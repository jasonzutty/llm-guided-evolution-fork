#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:3
#SBATCH -C "H200"
#SBATCH --mem 10G
#SBATCH -c 48
echo "Launching AIsurBL"
hostname
module load gcc/13.2.0
module load uv
source ~/.bashrc
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"
uv run python llm_crossover.py '/storage/ice1/2/5/mgullapalli6/llmGE/llm-guided-evolution-infrastructure/sota/Titanic/model.py' '/storage/ice1/2/5/mgullapalli6/llmGE/llm-guided-evolution-infrastructure/sota/Titanic/models/Menghao/model_x.py' '/storage/ice1/2/5/mgullapalli6/llmGE/llm-guided-evolution-infrastructure/sota/Titanic/models/Menghao/model_z.py'  --top_p 0.15   --temperature 0.1 --apply_quality_control 'True' --bit 8
