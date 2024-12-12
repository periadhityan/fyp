import torch
import torch.nn as nn
import dgl
import dgl. nn as dglnn

class GNN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, graph):
        super(GNN, self).__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(in_feats, hidden_size)
            for etype in graph.etypes
        })

        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(hidden_size, hidden_size)
            for etype in graph.etypes
        })

        self.fc = nn.Linear(hidden_size, out_feats)

    def forward(self, graph, inputs):
        h = {ntype: inputs[ntype] for ntype in graph.ntypes}

        h = self.conv1(graph, h)
        h = {k: torch.relu(v) for k, v in h.items()}

        h = self.conv1(graph, h)
        h = {k: torch.relu(v) for k, v in h.items()}

        with graph.local_Scope():
            for ntype in graph.ntypes:
                graph.nodes[ntype].data['h'] = h[ntype]
            hg = dgl.mean_nodes(graph, 'h')

        return self.fc(hg)