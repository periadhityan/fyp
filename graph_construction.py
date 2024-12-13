import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl
import torch
import json

def create_heterograph(json_file):

    with open(json_file, "r") as f:
        data = json.load(f)

    graph_data = {}

    for edge, node_pairs in data.items():
        source_type, edge_type, destination_type = edge.split("-")

        source = node_pairs[0]
        destination = node_pairs[1]

        graph_data[(source_type, edge_type, destination_type)] = (torch.tensor(source), torch.tensor(destination))

    hetero_graph = dgl.heterograph(graph_data)

    return hetero_graph
    


    




 