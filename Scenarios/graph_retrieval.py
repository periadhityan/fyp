from tqdm import tqdm
import os
import shutil

def main():
    output_dir = "/home/periadhityan/fyp/malicious_outputs"
    flurry_output_dir = "/home/periadhityan/flurry/output"
    i = 1
    for folder in tqdm(os.listdir(flurry_output_dir), desc="Copying Graphs", unit="folder"):
        folder_path = os.path.join(flurry_output_dir, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if "graph" in filename and filename.endswith(".json"):
                    src_path = os.path.join(folder_path, filename)
                    dest_path = os.path.join(output_dir, f"graph{i}.json")
                    shutil.copyfile(src_path, dest_path)
                    i += 1

if __name__ == '__main__':
    main()