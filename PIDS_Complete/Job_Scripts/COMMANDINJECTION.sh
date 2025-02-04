#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=q_ug1x16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=94G
#SBATCH --ntasks-per-node=20
#SBATCH --time=960
#SBATCH --job-name=CI
#SBATCH --output=Job_Outputs/output_%x_%j.out
#SBATCH --error=Job_Outputs/error_%x_%j.err
#SBATCH --chdir=/home/FYP/peri0006/fyp/PIDS_Complete

module load cuda/11.8
module load anaconda
source activate TestEnv

ATTACK=COMMANDINJECTION
FEATS=64
EPOCHS=10

python train.py $ATTACK $FEATS $EPOCHS None 1

for i in {2..16}; do
    python train.py $ATTACK $FEATS $EPOCHS Load $i
done

python test.py $ATTACK $FEATS