import subprocess
import os
import shutil
from tqdm import tqdm  # For progress bars

def flurry_webserver(input_scenario):
    """
    Executes the webserver script with the given input scenario.
    """
    p = subprocess.Popen(['python', 'webserver.py'], 
                         stdin=subprocess.PIPE, 
                         stdout=subprocess.PIPE, 
                         encoding='utf8')
    p.communicate(input=input_scenario)

def main():
    suffix = "1\n1\nf\nc"
    input_file = "mini_sample.txt"
    output_dir = "/home/periadhityan/fyp/benign_outputs"
    flurry_output_dir = "/home/periadhityan/flurry/output"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Read and process input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing Scenarios", unit="scenario"):
        line = line.strip()  # Remove trailing whitespace or newline
        print(f"Running Scenario: {line}")
        input_data = line + suffix
        flurry_webserver(input_data)

    # Traverse flurry output directory and copy relevant files
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

if __name__ == "__main__":
    main()