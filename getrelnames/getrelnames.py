from tqdm import tqdm
import os
import json

def main():
    folder1 = "Benign_Graphs"
    folder2 = "XSSSTORED_Graphs"

    unique_rel_names = set()

    for file in tqdm(os.listdir(folder1), desc="Gathering Relation Names", unit="Graphs"):
        if file.endswith(".json"):
            json_file = os.path.join(folder1, file)
            edges = get_etypes(json_file)
            unique_rel_names.update(edges)
    for file in tqdm(os.listdir(folder2), desc="Gathering Relation Names", unit="Graphs"):
        if file.endswith(".json"):
            json_file = os.path.join(folder2, file)
            edges = get_etypes(json_file)
            unique_rel_names.update(edges)

    print(len(unique_rel_names))
    print(unique_rel_names)

    for name in unique_rel_names:
        with(open(f'rel_names.txt', 'a')) as output:
            output.write((f'{name}\n'))


def get_etypes(file):

    with open(file, "r") as f:
        data = json.load(f)

    edges = []
    for edge, node_pairs in data.items():
        stype, etype, dsttype = edge.split("-")
        edges.append(etype)

    return edges
        

def test():
    file = open("rel_names.txt", "r")
    lines = [line.strip() for line in file.readlines()]
    print(len(lines))

test()