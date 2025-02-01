#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=q_ug1x16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=94G
#SBATCH --ntasks-per-node=20
#SBATCH --time=960
#SBATCH --job-name=XSSR_32_20
#SBATCH --output=Job_Outputs/output_%x_%j.out
#SBATCH --error=Job_Outputs/error_%x_%j.err
#SBATCH --chdir=/home/FYP/peri0006/fyp/PIDS

module load cuda/11.8
module load anaconda
source activate TestEnv

ATTACK=XSSREFLECTED
FEATS=32
EPOCHS=20

python 01_train_model.py $ATTACK $FEATS $EPOCHS None 1

for i in {2..16}; do
    python 01_train_model.py $ATTACK $FEATS $EPOCHS Load $i
done

python 02_test_model.py $ATTACK $FEATS $EPOCHS