import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from graph_creation import CreatingGraphs
from model import HeteroClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    malicious = "XSSREFLECTED_Train2"
    malicious_type = "malicious"

    malicious_graphs, malicious_labels = CreatingGraphs(malicious, malicious_type)

    file = open("rel_names.txt", "r")
    unique_rel_names = [line.strip() for line in file.readlines()]

    print("Making dataloader")
    dataset = list(zip(malicious_graphs, malicious_labels['labels']))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    print("Loading Model")
    model = HeteroClassifier(32, 32, 2, unique_rel_names)
    model.load_state_dict(torch.load("XSSREFLECTED_32_Feat.pth"))
    model.to(device)

    optimiser = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 20

    with(open(f'Results_XSSREFLECTED.txt', 'a')) as output:
        output.write((f'training set {malicious}\n'))

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

        with(open(f'Results_XSSREFLECTED.txt', 'a')) as output:
            output.write((f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}\n'))
            
        torch.cuda.empty_cache()

    torch.save(model, "XSSREFLECTED_32_Feat.pth")

def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels

if __name__ == '__main__':
    main()