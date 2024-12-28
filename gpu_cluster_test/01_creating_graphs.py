import dgl
import torch
import json
import os
from tqdm import tqdm

def main():
    benign = "Benign_Graphs"
    benign_type = "benign"
    malicious = "XSSSTORED_Graphs"
    malicious_type = "malicious"

    benign_graphs, benign_labels = CreatingGraphs(benign, benign_type)
    dgl.save_graphs('benign.bin', benign_graphs, benign_labels)

    malicous_graphs, malicious_labels = CreatingGraphs(malicious, malicious_type)
    dgl.save_graphs('xssstored.bin', malicous_graphs, malicious_labels)


def CreatingGraphs(graphs_folder, graphs_type):

    graphs = []

    for file in tqdm(os.listdir(graphs_folder), desc="Creating Graphs", unit="Graphs"):
        json_file = os.path.join(graphs_folder, file)
        g = create_graph(json_file)
        graphs.append(g)

    if graphs_type == "benign":
        labels = {'labels': torch.zeros(len(graphs))}
    else:
        labels = {'labels': torch.ones(len(graphs))}

    return graphs, labels

def create_graph(file):

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
        g.nodes[ntype].data['h'] = torch.randn(g.num_nodes(ntype), 32)

    return g 

if __name__ == '__main__':
    main()