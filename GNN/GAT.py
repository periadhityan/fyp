import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class HeteroGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, rel_names):
        super().__init__()
        self.layers = nn.ModuleDict({
            etype: dglnn.GATConv(in_feats, hidden_feats, num_heads, feat_drop=0.1, attn_drop=0.1, activation=F.elu, allow_zero_in_degree=True)
            for etype in rel_names
        })
        self.fc = nn.Linear(hidden_feats * num_heads, out_feats)

    def forward(self, g, inputs):
        h = {ntype: inputs[ntype] for ntype in g.ntypes}

        for rel in g.canonical_etypes:
            src_type, e_type, dst_type = rel
            if e_type in self.layers:  # Ensure relation exists in layers
                h_out = self.layers[e_type](g[e_type], h[src_type])

            # Check for shape mismatch and adjust
            if h_out.shape[0] != g.num_nodes(dst_type):
                print(f"Shape mismatch: {e_type} → Expected {g.num_nodes(dst_type)}, got {h_out.shape[0]}")
                h_out = h_out[: g.num_nodes(dst_type)]  # Trim extra rows or adjust as needed
                
                h[dst_type] = h_out  

        return {ntype: self.fc(h[ntype].mean(1)) for ntype in g.ntypes}

class HeteroGATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_heads, rel_names):
        super().__init__()
        self.gat = HeteroGAT(in_dim, hidden_dim, hidden_dim, num_heads, rel_names)
        self.classify = nn.Linear(hidden_dim, num_classes)

    def forward(self, g):
        inputs = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}
        h = self.gat(g, inputs)

        with g.local_scope():
            for ntype in g.ntypes:
                g.nodes[ntype].data['h'] = h[ntype]

            hg = torch.stack([dgl.mean_nodes(g, 'h', ntype) for ntype in g.ntypes]).sum(0)
            return self.classify(hg)
