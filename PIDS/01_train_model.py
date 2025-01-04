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
    graphs_folder = sys.argv[1]
    graphs_type = sys.argv[2]
    attack = sys.argv[3]
    feats = sys.argv[4]
    model_to_load = sys.arg[5]

    results_file = f"{attack}_results.txt"
    

    graphs, labels = CreatingGraphs(graphs_folder, graphs_type)

    file = open("rel_names.txt", "r")
    unique_rel_names = [line.strip() for line in file.readlines()]

    dataset = list(zip(graphs, labels['labels']))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    model = HeteroClassifier(feats, feats, 2, unique_rel_names)
    if model_to_load != "None":
        model.load_state_dict(torch.load(model_to_load))
    model.to(device)

    optimiser = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 20

    with(open(results_file, 'a')) as output:
        output.write((f'Training set {graphs_folder}\n'))

    print("Training starts here")
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
        
    torch.save(model.state_dict(), f"{attack}_{feats}.pth")

def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels

if __name__ == '__main__':
    main()