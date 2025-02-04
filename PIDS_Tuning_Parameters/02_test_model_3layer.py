import dgl
import torch
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from graph_creation import CreatingGraphs
from model_3layer import HeteroClassifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    benign_graphs = "BENIGN/Benign_Test"
    attack_type = sys.argv[1]
    feats = int(sys.argv[2])
    num_epochs = int(sys.argv[3])

    results_file = f"Results/{attack_type}_{feats}_3layer_results.txt"
    malicious_graphs = f"{attack_type}/{attack_type}_Test"
    model_to_load = f"Models/{attack_type}_{feats}_3layer.pth"

    benign_graphs, benign_labels = CreatingGraphs(benign_graphs, "Benign", feats)
    malicious_graphs, malicious_labels = CreatingGraphs(malicious_graphs, attack_type, feats)

    graphs = benign_graphs+malicious_graphs
    labels = torch.cat([benign_labels['labels'], malicious_labels['labels']])

    dataset = list(zip(graphs, labels))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    file = open("rel_names.txt", "r")
    unique_rel_names = [line.strip() for line in file.readlines()]
    
    model = HeteroClassifier(feats, feats, 2, unique_rel_names)
    model.load_state_dict(torch.load(model_to_load))
    model.to(device)

    model.eval()
    
    predictions = []
    labels = []

    with(open(results_file, 'a')) as output:
        output.write(f"\nEvaluating {attack_type} Model with {feats} node features and {num_epochs} epochs\n")

    with torch.no_grad():
        for graph, label in dataloader:
            graph = graph.to(device)
            
            logits = model(graph)

            prediction = torch.argmax(logits, dim=1)

            prediction = prediction.cpu()
            predictions.append(prediction)
            labels.append(label)

    report = classification_report(labels, predictions)
    with(open(results_file, 'a')) as output:
        output.write(report)

def custom_collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batch_labels = torch.stack(labels)

    return batched_graph, batch_labels

if __name__ == "__main__":
    main()