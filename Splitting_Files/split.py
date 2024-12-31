import os
import shutil
import random
from tqdm import tqdm

source = "XSSDOM_Graphs"
train1 = "XSSDOM_Train1"
train2 = "XSSDOM_Train2"
test = "XSSDOM_Test"

os.makedirs(train1, exist_ok=True)
os.makedirs(train2, exist_ok=True)
os.makedirs(test, exist_ok=True)

json_files = [f for f in os.listdir(source)]

random.shuffle(json_files)

train_ratio = 0.8
test_ratio = 0.2

train1_files = json_files[:800]
train2_files = json_files[800:1600]
test_files = json_files[1600:]

for file in tqdm(train1_files, desc="Creating Train Set", unit="Files"):
    shutil.move(os.path.join(source, file), os.path.join(train1, file))

for file in tqdm(train2_files, desc="Creating Train Set", unit="Files"):
    shutil.move(os.path.join(source, file), os.path.join(train2, file))

for file in tqdm(test_files, desc="Creating Test Set", unit="Files"):
    shutil.move(os.path.join(source, file), os.path.join(test, file))