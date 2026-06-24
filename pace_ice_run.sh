#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 16:00:00              		# Runtime in D-HH:MM
#SBATCH --mem 16G
#SBATCH -c 4                          # number of CPU cores
#SBATCH -N 1
echo "launching LLM Guided Evolution"
hostname
# module load anaconda3/2020.07 2021.11
module load cuda
export CUDA_VISIBLE_DEVICES=0
export SERVER_HOSTNAME=$(hostname)
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

uv run python run_improved.py titanic_test