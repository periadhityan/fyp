import os
import shutil
import random
from tqdm import tqdm

source = "BRUTEFORCE_Graphs"
train = "BRUTEFORCE_Train"
test = "BRUTEFORCE_Test"

json_files = [f for f in os.listdir(source)]

random.shuffle(json_files)

train_ratio = 0.8
test_ratio = 0.2

train_files = json_files[:1600]
test_files = json_files[1600:]

for file in tqdm(train_files, desc="Creating Train Set", unit="Files"):
    shutil.move(os.path.join(source, file), os.path.join(train, file))

for file in tqdm(test_files, desc="Creating Test Set", unit="Files"):
    shutil.move(os.path.join(source, file), os.path.join(test, file))