#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=360
#SBATCH --job-name=GNN
#SBATCH --output=job_outputs/output_%x%j.out
#SBATCH --error=job_outputs/error_%x_%j.err









module load cuda/11.8
module load anaconda
source activate TestEnv
python 02_graph_classifier.py