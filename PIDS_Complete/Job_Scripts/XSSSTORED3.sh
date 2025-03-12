#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=20
#SBATCH --time=360
#SBATCH --job-name=XSSS
#SBATCH --output=Job_Outputs/output_%x_%j.out
#SBATCH --error=Job_Outputs/error_%x_%j.err
#SBATCH --chdir=/home/FYP/peri0006/fyp/PIDS_Complete


module load cuda/11.8
module load anaconda
source activate TestEnv

ATTACK=XSSSTORED
FEATS=64
EPOCHS=10

for i in {1..15}; do
    python train.py $ATTACK $FEATS $EPOCHS Load $i
done
