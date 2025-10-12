#!/bin/bash
#SBATCH --job-name=IslandsController
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=16:00:00
#SBATCH --output=run_job_outputs/Report-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel

# This is the first argument passed to the script, which is the number of iterations
# Usage: sbatch pace_ice_island_controller.sbatch <number_of_iterations>
# Example: sbatch pace_ice_island_controller.sbatch 10
# This is a workaround for pace_ice time limits
COUNT=$1

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda/12

# LLM_Storage on PACE-ICE
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Change to the directory where the script is located (replace <username> with your actual username)
cd /home/hice1/<username>/scratch/llm-guided-evolution-Island-Migration

# Starts running Island Migration (replace <directory_name> with the directory you want to store the .pkl files of each generation in)
uv run python islands_wrapper.py <directory_name> --num_islands 1

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    echo "Launching next iteration with count: $NEXT_COUNT"
    uv run sbatch pace_ice_island_controller.sbatch "$NEXT_COUNT"
fi