import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl
import torch
import json
import time

json_files = [os.path.join('benign_outputs', file) for file in os.listdir('benign_outputs') if file.endswith('.json')]

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

    #print(f"Types of nodes: {hetero_graph.ntypes}")

    num_activity = hetero_graph.num_nodes('activity')
    num_agent= hetero_graph.num_nodes('agent')
    num_entity = hetero_graph.num_nodes('entity')

    #print("Creating one-hot encoded features")

    activity_feats = torch.eye(num_activity)
    agent_feats = torch.eye(num_agent)
    entity_feats = torch.eye(num_entity)

    node_features = {
        'activity': activity_feats,
        'agent': agent_feats,
        'entity': entity_feats
    }

    #print("Assigning featutes to nodes in graph")
    for ntype, features in node_features.items():
        hetero_graph.nodes[ntype].data['feat'] = features

    return hetero_graph


    




 