import dgl
import os
import torch 
import torch.nn as nn
import dgl.function as fn
import dgl.nn as dglnn
import torch.nn.functional as F
import json
import itertools
from tqdm import tqdm
os.environ["DGLBACKEND"] = "pytorch"

def main():
    
    message = create_graph("for_comparison/MESSAGE.json", 16, "MESSAGE")
    message2 = create_graph("for_comparison/MESSAGE2.json", 16, "MESSAGE2")

    submit = create_graph("for_comparison/SUBMIT.json", 16, "SUBMIT")
    submit2 = create_graph("for_comparison/SUBMIT2.json", 16, "SUBMIT2")

    query = create_graph("for_comparison/QUERY.json", 16, "QUERY")
    query2 = create_graph("for_comparison/QUERY2.json", 16, "QUERY2")

    ping = create_graph("for_comparison/PING.json", 16, "PING")
    ping2 = create_graph("for_comparison/PING2.json", 16, "PING2")

    databaseentry  = create_graph("for_comparison/DATABASEENTRY.json", 16, "DATABASEENTRY")
    databaseentry2  = create_graph("for_comparison/DATABASEENTRY2.json", 16, "DATABASEENTRY2")

    login = create_graph("for_comparison/LOGIN.json", 16, "LOGIN")
    login2 = create_graph("for_comparison/LOGIN2.json", 16, "LOGIN2")

    xssreflected = create_graph("for_comparison/XSSREFLECTED.json", 16, "XSSREFLECTED")
    xssreflected2 = create_graph("for_comparison/XSSREFLECTED2.json", 16, "XSSREFLECTED2")

    xssstored = create_graph("for_comparison/XSSSTORED.json", 16, "XSSSTORED")
    xssstored2 = create_graph("for_comparison/XSSSTORED2.json", 16, "XSSSTORED2")

    xssdom = create_graph("for_comparison/XSSDOM.json", 16, "XSSDOM")
    xssdom2 = create_graph("for_comparison/XSSDOM2.json", 16, "XSSDOM2")

    sqlinjection = create_graph("for_comparison/SQLINJECTION.json", 16, "SQLINJECTION")
    sqlinjection2 = create_graph("for_comparison/SQLINJECTION2.json", 16, "SQLINJECTION2")

    commandinjection = create_graph("for_comparison/COMMANDINJECTION.json", 16, "COMMANDINJECTION")
    commandinjection2 = create_graph("for_comparison/COMMANDINJECTION2.json", 16, "COMMANDINJECTION2")

    bruteforce = create_graph("for_comparison/BRUTEFORCE.json", 16, "BRUTEFORCE")
    bruteforce2 = create_graph("for_comparison/BRUTEFORCE2.json", 16, "BRUTEFORCE2")

    benign_for_comparison = [(message, message2), (submit, submit2), (query, query2), (ping, ping2), (databaseentry, databaseentry2), (login, login2)]
    malicious_for_comparison = [(xssstored, xssstored2), (xssreflected, xssreflected2), (xssdom, xssdom2), (sqlinjection, sqlinjection2), (commandinjection, commandinjection2), (bruteforce, bruteforce2)]
    counter_graphs_for_comparison = [(xssstored, message), (xssreflected, submit), (xssdom, query), (commandinjection, ping), (sqlinjection, databaseentry), (bruteforce, login)]

    for benign_graphs in benign_for_comparison:
        compare_raw_graphs(*benign_graphs)

    for malicious_graphs in malicious_for_comparison:
        compare_raw_graphs(*malicious_graphs)

    for graphs in counter_graphs_for_comparison:
        compare_raw_counter_graphs(*graphs)

    

def create_graph(file, feats, name):

    with open(file, "r") as f:
        data = json.load(f)

    graph_data = {}

    for edge, node_pairs in data.items():
        stype, etype, dsttype = edge.split("-")
        
        source = node_pairs[0]
        destination = node_pairs[1]

        graph_data[(stype, etype, dsttype)] = (torch.tensor(source), torch.tensor(destination))
    
    g = dgl.heterograph(graph_data)
        
    for ntype in g.ntypes:
        g.nodes[ntype].data['h'] = torch.randn(g.num_nodes(ntype), feats)

    for srctype, etype, dsttype in g.canonical_etypes:
        g.edges[(srctype, etype, dsttype)].data['feat'] = torch.randn(
            g.num_edges((srctype, etype, dsttype)), feats
        )
    

    return g, name 

def compare_raw_graphs(g1_data, g2_data):
        
        g1 = g1_data[0]
        g1_name = g1_data[1]

        g2 = g2_data[0]
        g2_name = g2_data[1]

        print(f"Comparing {g1_name} Graphs")

        g1_nodes = set()
        g2_nodes = set()
        g1_edges = set()
        g2_edges = set()

        for ntype in g1.ntypes:
            g1_nodes.update(g1.nodes(ntype).tolist())
        for ntype in g2.ntypes:
            g2_nodes.update(g2.nodes(ntype).tolist())

        for etype in g1.canonical_etypes:
            src, dst = g1.edges(etype=etype)
            g1_edges.update(zip(src.tolist(), dst.tolist(), itertools.repeat(etype)))
        for etype in g2.canonical_etypes:
            src, dst = g2.edges(etype=etype)
            g2_edges.update(zip(src.tolist(), dst.tolist(), itertools.repeat(etype)))

        node_overlap = len(g1_nodes & g2_nodes) / len(g1_nodes | g2_nodes) if g1_nodes | g2_nodes else 0
        edge_overlap = len(g1_edges & g2_edges) / len(g1_edges | g2_edges) if g1_edges | g2_edges else 0

        print(f"Node Overlap: {node_overlap*100:.2f}%")
        print(f"Edge Overlap: {edge_overlap*100:.2f}%")
        print()

def compare_raw_counter_graphs(g1_data, g2_data):
        g1 = g1_data[0]
        g1_name = g1_data[1]

        g2 = g2_data[0]
        g2_name = g2_data[1]

        print(f"Comparing {g1_name} and  {g2_name} Graphs")

        g1_nodes = set()
        g2_nodes = set()
        g1_edges = set()
        g2_edges = set()

        for ntype in g1.ntypes:
            g1_nodes.update(g1.nodes(ntype).tolist())
        for ntype in g2.ntypes:
            g2_nodes.update(g2.nodes(ntype).tolist())

        for etype in g1.canonical_etypes:
            src, dst = g1.edges(etype=etype)
            g1_edges.update(zip(src.tolist(), dst.tolist(), itertools.repeat(etype)))
        for etype in g2.canonical_etypes:
            src, dst = g2.edges(etype=etype)
            g2_edges.update(zip(src.tolist(), dst.tolist(), itertools.repeat(etype)))

        node_overlap = len(g1_nodes & g2_nodes) / len(g1_nodes | g2_nodes) if g1_nodes | g2_nodes else 0
        edge_overlap = len(g1_edges & g2_edges) / len(g1_edges | g2_edges) if g1_edges | g2_edges else 0

        print(f"Node Overlap: {node_overlap*100:.2f}%")
        print(f"Edge Overlap: {edge_overlap*100:.2f}%")
        print()


            

def multi_round_message_passing(graph, convolution, rounds):
    for _ in range(rounds):
        feature_dictionary = {ntype: graph.nodes[ntype].data['h'] for ntype in graph.ntypes}
        convolution.forward(feature_dictionary)


class GraphConvolution(nn.Module):
    def __init__(self, graph, input_features, hidden_features):
        super(GraphConvolution, self).__init__()
        self.graph = graph
        self.weight = nn.ModuleDict({
            f'{srctype}-{etype}-{dsttype}': nn.Linear(input_features[srctype], hidden_features[(srctype, etype, dsttype)])
            for srctype, etype, dsttype in graph.canonical_etypes
        })

    def forward(self, feature_dictionary):
        g = self.graph
        funcs = {}

        for srctype, etype, dsttype in g.canonical_etypes:
            key = f'{srctype}-{etype}-{dsttype}'
            Wh = self.weight[key](feature_dictionary[srctype])
            g.nodes[srctype].data['h'] = Wh
            funcs[(srctype, etype, dsttype)] = fn.copy_u(f'h', 'm'), fn.mean('m', 'h')

        g.multi_update_all(funcs, 'sum')

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
        
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')
    
    def forward(self, graph):

        inputs = {ntype: graph.nodes[ntype].data['h'] for ntype in graph.ntypes}
        
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)

        with graph.local_scope():
            for ntype in h:
                graph.nodes[ntype].data['h'] = h[ntype]

        return graph



if __name__ == "__main__":
    main()


