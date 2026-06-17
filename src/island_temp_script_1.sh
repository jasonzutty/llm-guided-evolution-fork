#!/bin/bash
#SBATCH --job-name=Islands_script
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=run_job_outputs/islands/Report_islands-%j.out
#SBATCH --gres=gpu:1
# #SBATCH -C intel
cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Run Python script
uv run python run_improved.py --checkpoints titanic_islands_run2/island_llama3_titanic-focused --global_path titanic_islands_run2/global_data --llm_model llama3 --prompt_group titanic/focused --llm_avail False
