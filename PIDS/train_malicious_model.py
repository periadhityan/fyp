import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from graph_creation import CreatingGraphs
from model import HeteroClassifier


device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cpu')

def main():
    malicious = "XSSSTORED_Train1"
    malicious_type = "malicious"

    malicious_graphs, malicious_labels = CreatingGraphs(malicious, malicious_type)

    file = open("rel_names.txt", "r")
    unique_rel_names = [line.strip() for line in file.readlines()]

    print("Making dataloader")
    dataset = list(zip(malicious_graphs, malicious_labels['labels']))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    print("Loading Model")
    model = HeteroClassifier(32, 32, 2, unique_rel_names)
    model.load_state_dict(torch.load("Benign_Model_32_Feat.pth"))
    model.to(device1)

    optimiser = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 20
    accumulation_steps =  4

    print("Training starts here")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (graph, label) in enumerate(dataloader):
            label = label.long()

            graph = graph.to(device1)
            label = label.to(device1)

            logits = model(graph)

            loss = loss_fn(logits, label)
            loss = loss/accumulation_steps
            
            loss.backward()

            if (i+1)% accumulation_steps == 0 or i == len(dataloader)-1:
                optimiser.step()
                optimiser.zero_grad()

            total_loss += loss.item() * accumulation_steps

        with(open(f'Results_XSSSTORED.txt', 'a')) as output:
            output.write((f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}\n'))


    torch.save(model, "XSSSTORED_32_Feat.pth")

def ensure_all_edge_types(graph, all_edge_types):
    for e_type in all_edge_types:
        if e_type not in graph.etypes:
            print(f"Graph has no {e_type}")
            # Add dummy edges for the missing edge type
            src, dst = torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
            graph.add_edges(src, dst, etype=e_type)
            print(f"{e_type} added")
    return graph

def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels

if __name__ == '__main__':
    main()