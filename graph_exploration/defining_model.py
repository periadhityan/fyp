import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import torch
from message_passing import HeteroGraphConvLayer
from torch.utils.data import DataLoader

def main():
    graphs, labels = dgl.load_graphs('after_mp.bin')
    labels = labels['labels']

    dataset = list(zip(graphs, labels))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    input_dim = 3 
    hidden_dim = 128
    output_dim = len(torch.unique(labels))

    model = GraphClassifier(input_dim, hidden_dim, output_dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batched_graph, batched_labels in dataloader:
            batched_graph = batched_graph.to(device)
            batched_labels = batched_labels.to(device)

            logits = model(batched_graph)
            loss = criterion(logits, batched_labels.long())

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

    # Create a DataLoader for the test set
    test_graphs, test_labels = dgl.load_graphs('after_mp.bin')
    test_labels = test_labels['labels']
    test_dataset = list(zip(test_graphs, test_labels))
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=custom_collate_fn)

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batched_graph, batched_labels in test_dataloader:
            batched_graph = batched_graph.to(device)
            batched_labels = batched_labels.to(device)

            logits = model(batched_graph)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == batched_labels).sum().item()
            total += len(batched_labels)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, readout='mean'):
        super(GraphClassifier, self).__init__()
        self.readout = readout
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, output_dim)

    def forward(self, batched_graph):
        with batched_graph.local_scope():
            graph_feats = 0

            if(self.readout=='mean'):
                for ntype in batched_graph.ntypes:
                    graph_feats += dgl.mean_nodes(batched_graph, 'h', ntype=ntype)
            elif(self.readout=='sum'):
                for ntype in batched_graph.ntypes:
                    graph_feats += dgl.sum_nodes(batched_graph, 'h', ntype=ntype)
            elif(self.readout=='max'):
                for ntype in batched_graph.ntypes:
                    graph_feats += dgl.max_nodes(batched_graph, 'h', ntype=ntype)
            else:
                raise ValueError(f"Unsupported readout method: {self.readout}")
            
            graph_feats = F.relu(self.fc(graph_feats))
            out = self.classify(graph_feats)

            return out

def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels


    
        
if __name__ == '__main__':
    main()