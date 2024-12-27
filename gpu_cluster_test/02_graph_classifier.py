import dgl
import torch
import numpy
from tqdm import tqdm
import dgl.function as fn
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl.nn as dglnn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    benign_graphs, benign_labels = dgl.load_graphs("benign.bin")
    malicious_graphs, malicious_labels = dgl.load_graphs("malicious.bin")

    graphs = benign_graphs+malicious_graphs
    labels = torch.cat([benign_labels['labels'], malicious_labels['labels']])

    dataset = list(zip(graphs, labels))
    

    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    unique_rel_names = set()

    for g in graphs:
        unique_rel_names.update(g.etypes)

    unique_rel_names = sorted(unique_rel_names)

    
    model = HeteroClassifier(32, 32, 2, unique_rel_names)
    model.to(device)

    optimiser = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for graph, label in train_dataloader:
            label = label.long()

            graph = graph.to(device)
            label = label.to(device)

            logits = model(graph)

            loss = loss_fn(logits, label)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        with(open("outputs.txt", 'a')) as output:
            output.write((f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}\n'))

        
    model.eval()
    
    predictions = []
    labels = []

    with torch.no_grad():
        for graph, label in test_dataloader:
            graph = graph.to(device)
            
            logits = model(graph)

            prediction = torch.argmax(logits, dim=1)

            prediction = prediction.cpu()
            predictions.append(prediction)
            labels.append(label)

    report = classification_report(labels, predictions, zero_division=1)
    with(open("outputs.txt", 'a')) as output:
        output.write(report)

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
        
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')
    
    def forward(self, graph, inputs):
        
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)

        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        inputs = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}

        h = self.rgcn(g, inputs)

        with g.local_scope():
            for ntype in g.ntypes:
                if ntype in h:
                    g.nodes[ntype].data['h'] = h[ntype]
                else:
                    continue
            # Calculate graph representation by average readout.

            hg = None

            for ntype in g.ntypes:
                if hg is None:
                    hg = dgl.mean_nodes(g, 'h', ntype=ntype)
                else:
                    hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                
            return self.classify(hg)
        
def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels

if __name__ == "__main__":
    main()