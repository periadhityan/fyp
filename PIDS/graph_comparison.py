import dgl
import torch 
import networkx as nx
import json
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    message = create_graph("for_comparison/MESSAGE.json", 0, "MESSAGE")
    message2 = create_graph("for_comparison/MESSAGE2.json", 0, "MESSAGE2")

    submit = create_graph("for_comparison/SUBMIT.json", 0, "SUBMIT")
    submit2 = create_graph("for_comparison/SUBMIT2.json", 0, "SUBMIT2")

    query = create_graph("for_comparison/QUERY.json", 0, "QUERY")
    query2 = create_graph("for_comparison/QUERY2.json", 0, "QUERY2")

    ping = create_graph("for_comparison/PING.json", 0, "PING")
    ping2 = create_graph("for_comparison/PING2.json", 0, "PING2")

    databaseentry  = create_graph("for_comparison/DATABASEENTRY.json", 0, "DATABASEENTRY")
    databaseentry2  = create_graph("for_comparison/DATABASEENTRY2.json", 0, "DATABASEENTRY2")

    login = create_graph("for_comparison/LOGIN.json", 0, "LOGIN")
    login2 = create_graph("for_comparison/LOGIN2.json", 0, "LOGIN2")

    xssreflected = create_graph("for_comparison/XSSREFLECTED.json", 0, "XSSREFLECTED")
    xssreflected2 = create_graph("for_comparison/XSSREFLECTED2.json", 0, "XSSREFLECTED2")

    xssstored = create_graph("for_comparison/XSSSTORED.json", 0, "XSSSTORED")
    xssstored2 = create_graph("for_comparison/XSSSTORED2.json", 0, "XSSSTORED2")

    xssdom = create_graph("for_comparison/XSSDOM.json", 0, "XSSDOM")
    xssdom2 = create_graph("for_comparison/XSSDOM2.json", 0, "XSSDOM2")

    sqlinjection = create_graph("for_comparison/SQLINJECTION.json", 0, "SQLINJECTION")
    sqlinjection2 = create_graph("for_comparison/SQLINJECTION2.json", 0, "SQLINJECTION2")

    commandinjection = create_graph("for_comparison/COMMANDINJECTION.json", 0, "COMMANDINJECTION")
    commandinjection2 = create_graph("for_comparison/COMMANDINJECTION2.json", 0, "COMMANDINJECTION2")

    bruteforce = create_graph("for_comparison/BRUTEFORCE.json", 0, "BRUTEFORCE")
    bruteforce2 = create_graph("for_comparison/BRUTEFORCE2.json", 0, "BRUTEFORCE2")

    benign_for_comparison = [(message, message2), (submit, submit2), (query, query2), (ping, ping2), (databaseentry, databaseentry2), (login, login2)]

    for type in benign_for_comparison:
        g1_data = type[0]
        g2_data = type[1]

        compare_for_comparison(g1_data, g2_data)

    malicious_for_comparison = [(xssstored, xssstored2), (xssreflected, xssreflected2), (xssdom, xssdom2), (sqlinjection, sqlinjection2), (commandinjection, commandinjection2), (bruteforce, bruteforce2)]
    for type in malicious_for_comparison:
        g1_data = type[0]
        g2_data = type[1]

        compare_for_comparison(g1_data, g2_data)

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

    return g, name 

def compare_for_comparison(g1_data, g2_data):
        
        g1 = g1_data[0]
        g1_name = g1_data[1]

        g2 = g2_data[0]
        g2_name = g2_data[1]

        print(f"Comparing {g1_name} and {g2_name} for_comparison")

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

        print("Node Overlap: ", node_overlap)
        print("Edge Overlap: ", edge_overlap)
        print()




if __name__ == "__main__":
    main()


