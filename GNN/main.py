import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from GraphCreation import CreatingGraphs
from MessagePassing import PassingMessages

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

if __name__ == '__main__':
    main()