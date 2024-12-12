from dgl.data import DGLDataset
from graph_construction import create_heterograph
import torch

class HGraphDataset(DGLDataset):
    def __init__(self, json_files):
        self.json_files = json_files
        super().__init__(name='hetero_graph_dataset')

    def process(self):
        self.graphs = [create_heterograph(self.json_files[file]) for file in self.json_files]
        self.labels = "Benign"

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    