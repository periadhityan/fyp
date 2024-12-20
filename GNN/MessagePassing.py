import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.function as fn
import torch.nn as nn
from tqdm import tqdm

def PassingMessages(graphs, rounds):
    
    for g in tqdm(graphs, desc="Message Passing", unit="Graphs"):
        for _ in range(rounds):
            input = {ntype: 32 for ntype in g.ntypes}
            hidden = {(srctype, etype, dsttype): 32 for srctype, etype, dsttype in g.canonical_etypes}
            convolution = GraphConvolution(g, input, hidden)
            feat_dict = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}

            convolution(feat_dict)

class GraphConvolution():
    def __init__(self, graph, input, hidden):
        super(GraphConvolution, self).__init__()
        self.graph = graph
        self.weight = nn.ModuleDict({
            f'{srctype}-{etype}-{dsttype}': nn.Linear(input[srctype], hidden[(srctype, etype, dsttype)])
            for srctype, etype, dsttype in graph.canonical_etypes
        })

    def forward(self, feat_dict):
        g = self.graph
        funcs = {}

        for srctype, etype, dsttype in g.canonical_etypes:
            key = f'{srctype}-{etype}-{dsttype}'
            Wh = self.weight[key](feat_dict[srctype])
            g.nodes[srctype].data[f'Wh_{etype}'] = Wh
            funcs[(srctype, etype, dsttype)] = (fn.copy_u(f'Wh_{etype}', 'm'), fn.mean('m', 'h'))

        g.multi_update_all(funcs, 'sum')