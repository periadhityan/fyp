from dataset import HGraphDataset
import os
import dgl
import torch


json_files = [os.path.join('benign_outputs', file) for file in os.listdir('benign_outputs') if file.endswith('.json')]

data = HGraphDataset(json_files)
graphs = []
labels = {'Label': []}

for graph, label in data:
    graphs.append(graph)
    labels['Label'].append(label)

labels['Label'] = torch.tensor(labels['Label'])

dgl.save_graphs('graphs.bin', graphs, labels)


