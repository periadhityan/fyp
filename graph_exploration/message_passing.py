import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from tqdm import tqdm

def main():
    graphs, labels = dgl.load_graphs('benign_graphs.bin')

    """for graph in graphs:
        for ntype in graph.ntypes:
            print(graph.nodes[ntype].data['h'].shape)
        for etype in graph.canonical_etypes:
            print(graph.edges[etype].data['v'].shape)"""

    num_rounds = 1

    """for ntype in g1.ntypes:
        print(f"Features for {ntype}:\n{g1.nodes[ntype].data['h']}")"""

    for g in tqdm(graphs, desc="Message Passing", unit="Graphs"):
        for _ in range(num_rounds):
                input_dims = {ntype: 3 for ntype in g.ntypes}
                hidden_dims = {(srctype, etype, dsttype): 3 for srctype, etype, dsttype in g.canonical_etypes}
                conv_layer = HeteroGraphConvLayer(g, input_dims, hidden_dims)
                feat_dict = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}
                updated_feats = conv_layer(feat_dict)

    """for ntype in g.ntypes:
        print(f"Updated features for {ntype}:\n{g.nodes[ntype].data['h']}")"""

    #Save graphs to another bin file
        
    dgl.save_graphs('after_mp.bin', graphs, labels)

    """input_dims = {ntype: 3 for ntype in g2.ntypes}
    hidden_dims = {(srctype, etype, dsttype): 3 for srctype, etype, dsttype in g2.canonical_etypes}

    num_layers = 3

    multi_layer_model = MultiLayerHetroGraphConv(g2, input_dims, hidden_dims, num_layers)
    feat_dict = {ntype: g2.nodes[ntype].data['h'] for ntype in g2.ntypes}
    updated_feats = multi_layer_model(feat_dict)

    for ntype in g2.ntype:
        print(f"Updated features for {ntype}:\n{g2.nodes[ntype].data['h']}")"""    



class HeteroGraphConvLayer(nn.Module):
    def __init__(self, graph, input_dims, hidden_dims):
        super(HeteroGraphConvLayer, self).__init__()
        self.graph = graph

        self.weight = nn.ModuleDict({
            f'{srctype}-{etype}-{dsttype}': nn.Linear(input_dims[srctype], hidden_dims[(srctype, etype, dsttype)])
            for srctype, etype, dsttype in graph.canonical_etypes
        })
    
    def forward(self, feat_dict):

        """

        Perform message passing on the hetero graph

        Args: 
            feat_dict(dict): A dictionary of input features for each node

        Returns:
            updated node features as a dictionary

        """

        G = self.graph
        funcs = {}

        for srctype, etype, dsttype in G.canonical_etypes:
            
            weight_key = f'{srctype}-{etype}-{dsttype}'

            Wh = self.weight[weight_key](feat_dict[srctype])

            G.nodes[srctype].data[f'Wh_{etype}'] = Wh

            funcs[(srctype, etype, dsttype)] = (fn.copy_u(f'Wh_{etype}', 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')

        updated_feats = {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}

        return updated_feats
    
class MultiLayerHetroGraphConv(nn.Module):
    def __init__(self, graph, input_dims, hidden_dims, num_layers):
        super(MultiLayerHetroGraphConv, self).__init__()
        self.layers = nn.ModuleList([
            HeteroGraphConvLayer(graph, input_dims if i==0 else hidden_dims, hidden_dims)
            for i in range(num_layers)
        ])

    def forward(self, feat_dict):
        for layer in self.layers:
            feat_dict = layer(feat_dict)
        return feat_dict
    

if __name__ == '__main__':
    main()