import subprocess
import os
import shutil
from tqdm import tqdm

def flurry_webserver(input_scenario):
    p = subprocess.Popen(['python', 'webserver.py'], stdin=subprocess.PIPE,
                                                    stdout=subprocess.PIPE,
                                                    encoding='utf8')
    p.communicate(input=input_scenario)


def main():
    suffix = "1\n1\nf\nc"
    with open("mini_sample.txt", 'r') as f:
        lines = f.readlines
        
    for line in lines:
        print("Running Scenario: {}".format(line))
        imput = line+suffix
        flurry_webserver(input)

    i = 1
    directory = "/home/periadhityan/flurry/output"

    for folder in os.listdir(directory):
        for filename in os.listdir(f"{directory}/{folder}"):
            if("graph" in filename and "json" in filename):
                shutil.copyfile(f"{directory}/{folder}/{filename}",
                f"/home/periadhityan/fyp/benign_outputs/graph{i}.json")
                i+=1

if __name__== "__main__":
    main()