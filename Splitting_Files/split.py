import os
import shutil
import random
from tqdm import tqdm

source = "Splitting_Files/Benign_Graphs"
train = "Splitting_Files/Benign_Train"
test = "Splitting_Files/Benign_Test"
validate = "Splitting_Files/Benign_Validate"

os.makedirs(train, exist_ok=True)
os.makedirs(test, exist_ok=True)
os.makedirs(validate, exist_ok=True)

json_files = [f for f in os.listdir(source)]

random.shuffle(json_files)

train_ratio = 0.8
test_ratio = 0.2

total_files = len(json_files)
train_count = int(total_files*train_ratio)
test_count = int(total_files*test_ratio)

train_files = json_files[:train_count]
test_files = json_files[train_count:]

for file in tqdm(train_files, desc="Creating Train Set", unit="Files"):
    shutil.move(os.path.join(source, file), os.path.join(train, file))

for file in tqdm(test_files, desc="Creating Test Set", unit="Files"):
    shutil.move(os.path.join(source, file), os.path.join(test, file))