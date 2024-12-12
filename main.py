from dataset import HGraphDataset
import os

json_files = [os.path.join('benign_outputs', file) for file in os.listdir('benign_outputs') if file.endswith('.json')]
dataset = HGraphDataset(json_files)

print(len(dataset))

