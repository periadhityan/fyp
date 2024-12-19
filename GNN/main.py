import os
os.environ["DGLBACKEND"] = "pytroch"

import dgl
import torch
from tqdm import tqdm

from GNN.GraphCreation import create_graph
from GNN.MessagePassing import GraphConvolution

def main():
    graphs_folder = ""
    graphs_type = ""

    benign_graphs = ""
    message_passing_rounds = 3

    """graphs, labels = CreatingGraphs(graphs_folder, graphs_type)
    dgl.save_graphs('benign.bin', graphs, labels)"""

    """graphs, labels = dgl.load_graphs(benign_graphs)
    PassingMessages(graphs, message_passing_rounds)
    dgl.save_graphs('benign.bin', graphs, labels)"""




def CreatingGraphs(graphs_folder, graphs_type):

    graphs = []

    for file in tqdm(os.listdir(graphs_folder), desc="Creating Graphs", unit="Graphs"):
        json_file = os.path.join(graphs_folder, file)
        g = create_graph(json_file)
        graphs.append(g)

    labels = {'labels': torch.zeros(len(graphs)) if graphs_type == "benign" else torch.ones(len(graphs))}

    return graphs, labels

def PassingMessages(graphs, rounds):
    
    for g in tqdm(graphs, desc="Message Passing", unit="Graphs"):
        for _ in range(rounds):
            input = {ntype: 32 for ntype in g.ntypes}
            hidden = {(srctype, etype, dsttype): 32 for srctype, etype, dsttype in g.canonical_etypes}
            convolution = GraphConvolution(g, input, hidden)
            feat_dict = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}

            convolution(feat_dict)

if __name__ == '__main__':
    main()