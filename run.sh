#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00              		# Runtime in D-HH:MM
#SBATCH --mem-per-gpu 16G
#SBATCH -n 1                          # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"

echo "launching LLM Guided Evolution"
hostname
# module load anaconda3/2020.07 2021.11
module load cuda

export CUDA_VISIBLE_DEVICES=0

# For this line create a file called geminikey.sh and in that file write 'export GEMINI_API_KEY="your-api-key-here(leave the double quotes, remove the single quotes)"'
source geminikey.sh

source .venv/bin/activate

python run_improved.py pointnet_test
