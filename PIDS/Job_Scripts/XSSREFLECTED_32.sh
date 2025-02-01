#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --ntasks-per-node=5
#SBATCH --time=300
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

first_job=$(sbatch --export=ATTACK=$ATTACK,FEATS=$FEATS,EPOCHS=$EPOCHS,LOAD_TYPE=None,SET_NUM=$i Train_Job.sh | awk '{print $4}')

for i in {2..16}; do
    next_job=$(sbatch --dependency=afterok:$first_job --export=ATTACK=$ATTACK,FEATS=$FEATS,EPOCHS=$EPOCHS,LOAD_TYPE=Load,SET_NUM=$i Train_Job.sh | awk '{print $4}')
    first_job=$next_job
done

sbatch --dependency=afterok:$first_job --export=ATTACK=$ATTACK,FEATS=$FEATS Test_Job.sh