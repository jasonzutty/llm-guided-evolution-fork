#!/bin/bash
#SBATCH --job-name=LLM_Island_llama3
#SBATCH -N1 --ntasks-per-node=32
#SBATCH --mem-per-gpu=32G
#SBATCH --time=16:00:00
#SBATCH --output=run_job_outputs/islands/Report_islands-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda/12

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Run Python script
uv run python run_improved.py test11/island_llama3 --global_path test11/global_data --llm_model llama3
