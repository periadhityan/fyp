from dgl.data import DGLDataset
from graph_construction import create_heterograph
import torch

class HGraphDataset(DGLDataset):
    def __init__(self, json_files):
        self.json_files = json_files
        super().__init__(name='hetero_graph_dataset')

    def process(self):
        self.graphs = [create_heterograph(self.json_files[i]) for i in range(len(self.json_files))]
        self.labels = torch.tensor([0 for _ in self.graphs])
        # Creating benign graph dataset

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    