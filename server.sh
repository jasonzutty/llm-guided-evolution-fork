#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/slurm-%j.out
echo "launching LLM Server"

# Optional chained submission count to work around walltime limits
COUNT=${1:-1}
SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER:-1}

hostname

module load cuda
module load uv

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME (count=$COUNT)"

# Submit the paired island-controller job from here so the two stay in sync
echo "Submitting island controller (count=$COUNT)"
sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"

uv run python -m uvicorn server:app --host $SERVER_HOSTNAME --port 8137 --workers 1
