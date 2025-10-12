#!/bin/bash
#SBATCH --job-name=IslandsController
#SBATCH -t 30-00:00
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --output=run_job_outputs/Report-%j.out

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda

# LLM_Storage on PACE-ICE
export HF_HOME=

# Change to the directory where the script is located (replace <username> with your actual username)
cd /home/<username>/llm-guided-evolution-Island-Migration

# Starts running Island Migration (replace <directory_name> with the directory you want to store the .pkl files of each generation in)
uv run python islands_wrapper.py <directory_name> --num_islands 2
