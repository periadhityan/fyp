import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
        
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')
        
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')
    
    def forward(self, graph, inputs):
        
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)

        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        inputs = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}

        h = self.rgcn(g, inputs)

        with g.local_scope():
            for ntype in g.ntypes:
                if ntype in h:
                    g.nodes[ntype].data['h'] = h[ntype]
                else:
                    continue
            # Calculate graph representation by average readout.

            hg = None

            for ntype in g.ntypes:
                if hg is None:
                    hg = dgl.mean_nodes(g, 'h', ntype=ntype)
                else:
                    hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                
            return self.classify(hg)