import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from GraphCreation import CreatingGraphs
from MessagePassing import PassingMessages
from GraphClassifier import HetroClassifier

def main():
    graphs_folder = "Benign_Graphs\mini"
    graphs_type = "benign"

    benign_graphs = ""
    message_passing_rounds = 3


    graphs, labels = CreatingGraphs(graphs_folder, graphs_type)
    dgl.save_graphs(f'mini.bin', graphs, labels)

    graphs, labels = dgl.load_graphs(benign_graphs)
    PassingMessages(graphs, message_passing_rounds)
    dgl.save_graphs('benign.bin', graphs, labels)

    graphs, labels = dgl.load_graphs('mini.bin')
    labels = labels['labels']

    rel_names = graphs[0].canonical_etypes
    in_dim = 32
    hidden_dim = 32
    n_classes = 2

    model = HetroClassifier(in_dim, hidden_dim, n_classes, rel_names)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0.0
        for i, graph in enumerate(graphs):
            label = labels[i]
            
            assert 'h' in graph.ndata, f"Graph {i} does not have 'h' in ndata"

            logits = model(graph)

            loss = loss_fn(logits, label)
            total_loss += loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            print(f"Epoch {epoch + 1}/10, Graph {i+1}/{len(graphs)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss/len(graphs)
        print(f"Epoch {epoch+1}/{10}, Avg Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    main()