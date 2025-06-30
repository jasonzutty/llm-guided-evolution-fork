#!/bin/bash
#SBATCH --job-name=test_coco2017
#SBATCH -t 8:00:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu:2
#SBATCH --mem 16G
#SBATCH -c 1                    # number of CPU cores
# All of the above settings are configured for running on the PACE-ICE HPC

echo "launching test_coco2017"
hostname

module load cuda/12.6.1

conda deactivate
source .venv/bin/activate

echo "--- DEBUGGING PYTHON ENVIRONMENT ---"
which python
echo "--- END DEBUGGING ---"

uv run tests/sota/test_coco2017.py