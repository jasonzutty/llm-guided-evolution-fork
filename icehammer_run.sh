#!/bin/bash
#SBATCH --job-name=LLM_Island
#SBATCH -t 10-00:00
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --mem 80G

echo "launching LLM Guided Evolution"
hostname
module load cuda
export CUDA_VISIBLE_DEVICES=0

export HF_HOME=

uv run python run_improved.py test --global_path test/global_data --llm_model qwen25