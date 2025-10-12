# ROOT DIR in PACE ICE, change <username> to your actual username
ROOT_DIR = "/home/hice1/<username>/scratch/llm-guided-evolution-Island-Migration/"

# All GPUs available on PACE ICE (in order of performance)
# LLM_GPU = 'H200|H100|A100-80GB|A100-40GB|A40|RTX6000|V100-32GB|V100-16GB'
LLM_GPU = 'H200|H100'

#: Template script for submitting job for evaluation
PYTHON_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=evaluateGene
#SBATCH --time=06:00:00
#SBATCH -N1 --ntasks-per-node=32
#SBATCH --output=run_job_outputs/evaluation/slurm-%j.out

#SBATCH -G 1
#SBATCH -C "{}"
#SBATCH --mem-per-gpu 80G


echo "Launching AIsurBL"
hostname
# Load GCC version 9.2.0
# module load gcc/13.2.0
module load cuda/12

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export MKL_THREADING_LAYER=GNU

# Run Python script
uv run {}
"""

#: Template script for submitting a prompt to the LLM
LLM_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={}
#SBATCH --time=03:00:00
#SBATCH -N1 --ntasks-per-node=32
#SBATCH --output=run_job_outputs/evolution/slurm-%j.out

#SBATCH -G 1
#SBATCH -C "{}"
#SBATCH --mem-per-gpu 80G


echo "Launching AIsurBL"
hostname

module load cuda/12

CUDA_LAUNCH_BLOCKING=1

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Run Python script
uv run {}
"""

#: Template script for submitting an island run
ISLANDS_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=LLM_Island_{}
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
uv run python run_improved.py {} --global_path {} --llm_model {}
"""