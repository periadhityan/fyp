#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=4
#SBATCH --time=360
#SBATCH --job-name=GNN
#SBATCH --output=job_outputs/output_%x%j.out
#SBATCH --error=job_outputs/error_%x_%j.err

module load cuda/11.8
module load anaconda
source activate TestEnv
python 03_GNN.py