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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    benign = "Benign_Graphs"
    benign_type = "benign"
    malicious = "XSSSTORED_Graphs"
    malicious_type = "malicious"

    benign_graphs, benign_labels = CreatingGraphs(benign, benign_type)
    malicious_graphs, malicious_labels = CreatingGraphs(malicious, malicious_type)

    #dgl.save_graphs('benign.bin', benign_graphs, benign_labels)
    #dgl.save_graphs('xssstored.bin', malicous_graphs, malicious_labels)

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

        with(open(f'Results.txt', 'a')) as output:
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
    with(open('Results.txt', 'a')) as output:
        output.write(report)



def CreatingGraphs(graphs_folder, graphs_type):

    graphs = []

    for file in tqdm(os.listdir(graphs_folder), desc="Creating Graphs", unit="Graphs"):
        json_file = os.path.join(graphs_folder, file)
        g = create_graph(json_file)
        graphs.append(g)

    if graphs_type == "benign":
        labels = {'labels': torch.zeros(len(graphs))}
    else:
        labels = {'labels': torch.ones(len(graphs))}

    return graphs, labels

def create_graph(file):

    with open(file, "r") as f:
        data = json.load(f)

    graph_data = {}

    for edge, node_pairs in data.items():
        stype, etype, dsttype = edge.split("-")
        
        source = node_pairs[0]
        destination = node_pairs[1]

        graph_data[(stype, etype, dsttype)] = (torch.tensor(source), torch.tensor(destination))
    
    g = dgl.heterograph(graph_data)
    g.to(device)
    
    for ntype in g.ntypes:
        g.nodes[ntype].data['h'] = torch.randn(g.num_nodes(ntype), 32)

    return g 

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