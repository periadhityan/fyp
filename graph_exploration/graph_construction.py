import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl
import torch
import json

from dgl.data import DGLDataset
from torch.utils.data import DataLoader
import dgl.function as fn
import torch.nn as nn

from tqdm import tqdm

def main():

    graphs = []

    for file in tqdm(os.listdir("benign_outputs"), desc="Creating DGL graphs", unit="Graphs"):
        graph_file = os.path.join("benign_outputs", file)
        g = create_heterograph(graph_file)
        graphs.append(g)

    for g in tqdm(graphs, desc="Creating Node and Edge features", unit="Graphs"):
        #print("\nCreating Node Features for Graph\n")  
        for ntype in g.ntypes:
            #print(f"Number of nodes of type {ntype}: {g.num_nodes(ntype)}")
            g.nodes[ntype].data['h'] = torch.randn(g.num_nodes(ntype), 3)

        #print("\nCreating Edge Features for Graph\n")
        for etype in g.canonical_etypes:
            #print(f"Number of Edges for type {etype}: {g.num_edges(etype)}")
            g.edges[etype].data['v'] = torch.randn(g.num_edges(etype), 3)

    

    #Creating a dataset of graphs
    dataset = ProvenanceDataset(graphs, 'benign')

    #Saving graphs
    labels = {'labels': torch.zeros(len(dataset))}
    dgl.save_graphs('benign_graphs.bin', graphs, labels)



"""
    print("\nExploring Graphs with DGL")
    print("\nThese are the details for the graph as below")
    print(f"Number of Nodes: {g.num_nodes()}")
    print(f"Number of edges: {g.num_edges()}")

    for ntype in g.ntypes:
        print(f"Node type {ntype} has \t\t{g.nodes[ntype].data['feat'].shape[1]} features")

    for etype in g.canonical_etypes:
        print(f"Edge type {etype} has \t\t{g.edges[etype].data['feat'].shape[1]} features")
    
    for batch_graph, batch_labels in dataloader:
        print("Batched Graph: ", batch_graph)
        print("Batched Labels: ", batch_labels)

"""    

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


class ProvenanceDataset(DGLDataset):
    def __init__(self, graphs, label):
        super().__init__(name='provenance')
        self.graphs = graphs
        if label == 'benign':
            self.labels = torch.tensor([0 for _ in graphs])
        else:
            self.labels = torch.tensor([1 for _ in graphs])

    def process(self):
        pass

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.graphs)
    
def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels

if __name__ == '__main__':
    main()