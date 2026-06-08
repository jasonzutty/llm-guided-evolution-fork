#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/slurm-%j.out
echo "launching LLM Server"

# Optional chained submission count to work around walltime limits
COUNT=${1:-1}
SERVER_BACKEND=${2:-${LLMGE_SERVER_BACKEND:-vllm}}

hostname

module load cuda
module load uv

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
export XDG_CACHE_HOME="$UV_CACHE_DIR/xdg"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME (count=$COUNT, backend=$SERVER_BACKEND)"

# Submit the paired island-controller job from here so the two stay in sync
echo "Submitting island controller (count=$COUNT)"
sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"

case "$SERVER_BACKEND" in
    vllm)
        uv run --no-project --with "vllm>=0.8.5" --with fastapi --with uvicorn python -m uvicorn server_vllm:app --host $SERVER_HOSTNAME --port 2244 --workers 1
        ;;
    normal|transformers|baseline)
        uv run python -m uvicorn server:app --host $SERVER_HOSTNAME --port 2244 --workers 1
        ;;
    *)
        echo "Unknown LLM server backend '$SERVER_BACKEND'. Use 'vllm' or 'normal'." >&2
        exit 2
        ;;
esac
