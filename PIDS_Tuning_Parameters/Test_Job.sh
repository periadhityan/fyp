#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --ntasks-per-node=15
#SBATCH --time=60
#SBATCH --job-name=GNNTest
#SBATCH --output=Job_Outputs/output_%x%j.out
#SBATCH --error=Job_Outputs/error_%x_%j.err
#SBATCH --chdir=/home/FYP/peri0006/fyp/PIDS

module load cuda/11.8
module load anaconda
source activate TestEnv
python 02_test_model.py $ATTACK $FEATS