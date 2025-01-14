from tqdm import tqdm
import os
import json

def main():
    folder1 = "Benign_Graphs"
    folder2 = "XSSSTORED_Graphs"
    folder3 = "XSSREFLECTED_Graphs"
    folder4 = "BRUTEFORCE_Graphs"
    folder5 = "COMMANDINJECTION_Graphs"
    folder6 = "SQLINJECTION_Graphs"
    folder7 = "XSSDOM_Graphs"

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
    for file in tqdm(os.listdir(folder3), desc="Gathering Relation Names", unit="Graphs"):
        if file.endswith(".json"):
            json_file = os.path.join(folder3, file)
            edges = get_etypes(json_file)
            unique_rel_names.update(edges)
    for file in tqdm(os.listdir(folder4), desc="Gathering Relation Names", unit="Graphs"):
        if file.endswith(".json"):
            json_file = os.path.join(folder4, file)
            edges = get_etypes(json_file)
            unique_rel_names.update(edges)
    for file in tqdm(os.listdir(folder5), desc="Gathering Relation Names", unit="Graphs"):
        if file.endswith(".json"):
            json_file = os.path.join(folder5, file)
            edges = get_etypes(json_file)
            unique_rel_names.update(edges)
    for file in tqdm(os.listdir(folder6), desc="Gathering Relation Names", unit="Graphs"):
        if file.endswith(".json"):
            json_file = os.path.join(folder6, file)
            edges = get_etypes(json_file)
            unique_rel_names.update(edges)
    for file in tqdm(os.listdir(folder7), desc="Gathering Relation Names", unit="Graphs"):
        if file.endswith(".json"):
            json_file = os.path.join(folder7, file)
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
        

main()