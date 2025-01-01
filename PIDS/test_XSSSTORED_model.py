import dgl
import torch
import json
import os
import dgl.function as fn
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl.nn as dglnn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from graph_creation import CreatingGraphs
from model import HeteroClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    benign = "Benign_Test"
    benign_type = "benign"
    malicious = "XSSSTORED_Test"
    malicious_type = "malicious"

    benign_graphs, benign_labels = CreatingGraphs(benign, benign_type)
    malicious_graphs, malicious_labels = CreatingGraphs(malicious, malicious_type)

    graphs = benign_graphs+malicious_graphs
    labels = torch.cat([benign_labels['labels'], malicious_labels['labels']])

    dataset = list(zip(graphs, labels))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    file = open("rel_names.txt", "r")
    unique_rel_names = [line.strip() for line in file.readlines()]
    
    model = HeteroClassifier(32, 32, 2, unique_rel_names)
    model.load_state_dict(torch.load("XSSSTORED_32_Feat.pth"))
    model.to(device)

        
    model.eval()
    
    predictions = []
    labels = []

    with torch.no_grad():
        for graph, label in dataloader:
            graph = graph.to(device)
            
            logits = model(graph)

            prediction = torch.argmax(logits, dim=1)

            prediction = prediction.cpu()
            predictions.append(prediction)
            labels.append(label)

    report = classification_report(labels, predictions, zero_division=1)
    with(open(f'{malicious}_Results.txt', 'a')) as output:
        output.write(report)

def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels

if __name__ == "__main__":
    main()