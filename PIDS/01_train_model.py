import dgl
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from graph_creation import CreatingGraphs
from model import HeteroClassifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    attack_type = sys.argv[1]
    feats = int(sys.argv[2])

    malicious_graphs_folder = f"{attack_type}/{attack_type}_Train"

    results_file = f"Results/{attack_type}_{feats}_results.txt"
    
    malicious_graphs, malicious_labels = CreatingGraphs(malicious_graphs_folder, attack_type, feats)
    benign_graphs, benign_labels = CreatingGraphs("BENIGN/Benign_Train", "Benign")

    graphs = benign_graphs+malicious_graphs
    labels = torch.cat([benign_labels['labels'], malicious_labels['labels']])

    file = open("rel_names.txt", "r")
    unique_rel_names = [line.strip() for line in file.readlines()]

    dataset = list(zip(graphs, labels['labels']))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    model = HeteroClassifier(feats, feats, 2, unique_rel_names)
    model.to(device)

    optimiser = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 10

    with(open(results_file, 'a')) as output:
        output.write((f'Training with {attack_type} Graphs\n'))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for graph, label in dataloader:
            label = label.long()

            graph = graph.to(device)
            label = label.to(device)

            logits = model(graph)

            loss = loss_fn(logits, label)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        with(open(results_file, 'a')) as output:
            output.write((f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}\n'))
            
        torch.cuda.empty_cache()

    with(open(results_file, 'a')) as output:
        output.write('\n')
        
    torch.save(model.state_dict(), f"Models/{attack_type}_{feats}.pth")

def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels

if __name__ == '__main__':
    main()