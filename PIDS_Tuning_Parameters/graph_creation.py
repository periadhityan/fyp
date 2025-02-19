import dgl
import torch
import json
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def CreatingGraphs(graphs_folder, graphs_type, feats):

    graphs = []

    for file in tqdm(os.listdir(graphs_folder), desc=f"Creating {graphs_type} Graphs", unit="Graphs"):
        if file.endswith(".json"):
            json_file = os.path.join(graphs_folder, file)
            g = create_graph(json_file, feats)
            graphs.append(g)

    if graphs_type == "Benign":
        labels = {'labels': torch.zeros(len(graphs))}
    else:
        labels = {'labels': torch.ones(len(graphs))}

    return graphs, labels

def create_graph(file, feats):

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

    return g 