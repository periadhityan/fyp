#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=20
#SBATCH --time=360
#SBATCH --job-name=MalTrain
#SBATCH --output=Job_Outputs/output_%x%j.out
#SBATCH --error=Job_Outputs/error_%x_%j.err

module load cuda/11.8
module load anaconda
source activate TestEnv
python 01_train_model.py XSSREFLECTED/XSSREFLECTED_Train2 malicious XSSREFLECTED 64 Models/XSSREFLECTED_64.pth