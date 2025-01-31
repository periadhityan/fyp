#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=q_ug1x16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=94G
#SBATCH --ntasks-per-node=20
#SBATCH --time=960
#SBATCH --job-name=GNNTrain
#SBATCH --output=Job_Outputs/output_%x_%j.out
#SBATCH --error=Job_Outputs/error_%x_%j.err
#SBATCH --chdir=/home/FYP/peri0006/fyp/PIDS

module load cuda/11.8
module load anaconda
source activate TestEnv

ATTACK=XSSREFLECTED
FEATS=32
EPOCHS=10

# Run jobs sequentially with Load values from 1 to 16
for i in {1..16}; do
    if [$i eq 1]; then
        sbatch --export=ATTACK=$ATTACK,FEATS=$FEATS,EPOCHS=$EPOCHS,LOAD_TYPE=None,SET_NUM=1 Train_Job.sh
        sleep 1

    else
        sbatch --export=LOAD_TYPE=Load,SET_NUM=$i Train_Job.sh
        sleep 1
    fi
done

sbatch --export=ATTACK=$ATTACK,FEATS=$FEATS Test_Job.sh