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
    
    message = create_graph("for_comparison/MESSAGE.json", 64, "MESSAGE")
    message2 = create_graph("for_comparison/MESSAGE2.json", 64, "MESSAGE2")

    submit = create_graph("for_comparison/SUBMIT.json", 64, "SUBMIT")
    submit2 = create_graph("for_comparison/SUBMIT2.json", 64, "SUBMIT2")

    query = create_graph("for_comparison/QUERY.json", 64, "QUERY")
    query2 = create_graph("for_comparison/QUERY2.json", 64, "QUERY2")

    ping = create_graph("for_comparison/PING.json", 64, "PING")
    ping2 = create_graph("for_comparison/PING2.json", 64, "PING2")

    databaseentry  = create_graph("for_comparison/DATABASEENTRY.json", 64, "DATABASEENTRY")
    databaseentry2  = create_graph("for_comparison/DATABASEENTRY2.json", 64, "DATABASEENTRY2")

    login = create_graph("for_comparison/LOGIN.json", 64, "LOGIN")
    login2 = create_graph("for_comparison/LOGIN2.json", 64, "LOGIN2")

    xssreflected = create_graph("for_comparison/XSSREFLECTED.json", 64, "XSSREFLECTED")
    xssreflected2 = create_graph("for_comparison/XSSREFLECTED2.json", 64, "XSSREFLECTED2")

    xssstored = create_graph("for_comparison/XSSSTORED.json", 64, "XSSSTORED")
    xssstored2 = create_graph("for_comparison/XSSSTORED2.json", 64, "XSSSTORED2")

    xssdom = create_graph("for_comparison/XSSDOM.json", 64, "XSSDOM")
    xssdom2 = create_graph("for_comparison/XSSDOM2.json", 64, "XSSDOM2")

    sqlinjection = create_graph("for_comparison/SQLINJECTION.json", 64, "SQLINJECTION")
    sqlinjection2 = create_graph("for_comparison/SQLINJECTION2.json", 64, "SQLINJECTION2")

    commandinjection = create_graph("for_comparison/COMMANDINJECTION.json", 64, "COMMANDINJECTION")
    commandinjection2 = create_graph("for_comparison/COMMANDINJECTION2.json", 64, "COMMANDINJECTION2")

    bruteforce = create_graph("for_comparison/BRUTEFORCE.json", 64, "BRUTEFORCE")
    bruteforce2 = create_graph("for_comparison/BRUTEFORCE2.json", 64, "BRUTEFORCE2")

    all_graphs = [message, message2, submit, submit2, query, query2, ping, ping2, databaseentry, databaseentry2, login, login2,
                  xssstored, xssstored2, xssreflected, xssstored2, xssdom, xssdom2, commandinjection, commandinjection2, sqlinjection, sqlinjection2, bruteforce, bruteforce2]
    benign_for_comparison = [(message, message2), (submit, submit2), (query, query2), (ping, ping2), (databaseentry, databaseentry2), (login, login2)]
    malicious_for_comparison = [(xssstored, xssstored2), (xssreflected, xssreflected2), (xssdom, xssdom2), (sqlinjection, sqlinjection2), (commandinjection, commandinjection2), (bruteforce, bruteforce2)]
    counter_graphs_for_comparison = [(xssstored, message), (xssreflected, submit), (xssdom, query), (commandinjection, ping), (sqlinjection, databaseentry), (bruteforce, login)]


    for benign_graphs in benign_for_comparison:
        compar_same_type_node_features(*benign_graphs)
              

    for malicious_graphs in malicious_for_comparison:
        compar_same_type_node_features(*malicious_graphs)

    for graphs in counter_graphs_for_comparison:
        compare_diff_type_node_features(*graphs)
        
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

    file = open("rel_names.txt", "r")
    unique_rel_names = [line.strip() for line in file.readlines()]
    
    rgcn = RGCN(64, 64, 64, unique_rel_names)

    g = rgcn(g)

    return g, name 

def compar_same_type_node_features(g1_data, g2_data):
    g1, g1_name = g1_data
    g2, g2_name = g2_data

    print(f"Comparing node features of {g1_name} Graphs")

    for ntype in g1.ntypes:
        if ntype in g2.ntypes:
            features_g1 = g1.nodes[ntype].data['h']
            features_g2 = g2.nodes[ntype].data['h']

            min_nodes = min(features_g1.size(0), features_g2.size(0))
            #similarity = torch.cosine_similarity(features_g1[:min_nodes], features_g2[:min_nodes], dim=1).mean()
            difference = torch.norm(features_g1[:min_nodes] - features_g2[:min_nodes], dim=1).mean()
            print(f"Feature difference for node type '{ntype}': {difference.item():.4f}")
        else: 
            (f"{ntype} not found in {g1_name}")
    print()

def compare_diff_type_node_features(g1_data, g2_data):
    g1, g1_name = g1_data
    g2, g2_name = g2_data

    print(f"Comparing node features of {g1_name} and {g2_name} Graphs")

    for ntype in g1.ntypes:
        if ntype in g2.ntypes:
            features_g1 = g1.nodes[ntype].data['h']
            features_g2 = g2.nodes[ntype].data['h']

            min_nodes = min(features_g1.size(0), features_g2.size(0))
            #similarity = torch.cosine_similarity(features_g1[:min_nodes], features_g2[:min_nodes], dim=1).mean()
            difference = torch.norm(features_g1[:min_nodes] - features_g2[:min_nodes], dim=1).mean()
            print(f"Feature difference for node type '{ntype}': {difference.item():.4f}")
        else: 
            (f"{ntype} not found in {g1_name}")
    print()

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
    
    def forward(self, graph):

        inputs = {ntype: graph.nodes[ntype].data['h'] for ntype in graph.ntypes}
        
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)

        for ntype in h:
            graph.nodes[ntype].data['h'] = h[ntype]

        return graph
    
if __name__ == "__main__":
    main()