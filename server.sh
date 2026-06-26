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

# Backend selection:
# 1) Positional argument $2
# 2) Env var LLMGE_SERVER_BACKEND
# 3) Python constants.LLM_SERVER_BACKEND (derived from USE_VLLM)
SERVER_BACKEND=${2:-${LLMGE_SERVER_BACKEND:-vllm}}

# vLLM package (only used if SERVER_BACKEND=vllm)
VLLM_PACKAGE=${VLLM_PACKAGE:-vllm==0.8.5}

hostname
module load cuda
module load uv

# Respect Slurm's GPU assignment. If Slurm did not set CUDA_VISIBLE_DEVICES,
# fall back to the two local device ordinals requested by this job.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Cache dirs (used for both backends)
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
export XDG_CACHE_HOME="$UV_CACHE_DIR/xdg"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export FLASHINFER_CACHE_DIR="$XDG_CACHE_HOME/flashinfer"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_CACHE_DIR"

# Prevent transformers from importing TensorFlow (vLLM doesn't need it)
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3

echo "Using UV cache: $UV_CACHE_DIR"
echo "Using XDG cache: $XDG_CACHE_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# vLLM-specific env: only meaningful if SERVER_BACKEND=vllm
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-true}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "VLLM_DISABLE_CUSTOM_ALL_REDUCE=$VLLM_DISABLE_CUSTOM_ALL_REDUCE"

# FlashInfer tuning (again, only relevant if vLLM path is used)
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "FLASHINFER_CACHE_DIR=$FLASHINFER_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)
HOSTNAME_FILE=$(pwd)"/hostname.log"
echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"

# Load balancing configuration
export LLMGE_USE_LOAD_BALANCING=${LLMGE_USE_LOAD_BALANCING:-false}
export SERVER_REGISTRY_FILE=${SERVER_REGISTRY_FILE:-/storage/ice1/2/5/mgullapalli6/1/llm-guided-evolution-infrastructure/servers.json}
export LOAD_BALANCER_PORT=${LOAD_BALANCER_PORT:-9000}

if [ "$LLMGE_USE_LOAD_BALANCING" = "true" ]; then
    echo "Load balancing ENABLED"
    echo "  Registry file: $SERVER_REGISTRY_FILE"
    echo "  Load balancer port: $LOAD_BALANCER_PORT"

    # Initialize registry file if it doesn't exist
    if [ ! -f "$SERVER_REGISTRY_FILE" ]; then
        echo "Creating empty server registry: $SERVER_REGISTRY_FILE"
        echo '{"servers": []}' > "$SERVER_REGISTRY_FILE"
    fi
else
    echo "Load balancing DISABLED"
fi

echo "Starting LLM server on host: $SERVER_HOSTNAME (count=$COUNT, backend=$SERVER_BACKEND)"
echo "Using vLLM package: $VLLM_PACKAGE"

# Log the island controller setting for debugging
echo "SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER:-<not set>}"

# Default behavior: START island controller unless explicitly disabled
if [ "${SUBMIT_ISLAND_CONTROLLER:-1}" = "1" ]; then
    # Submit the paired island-controller job from here so the two stay in sync.
    echo "Submitting island controller (count=$COUNT)"
    sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"
else
    echo "Skipping island controller submission (SUBMIT_ISLAND_CONTROLLER=${SUBMIT_ISLAND_CONTROLLER})"
fi

case "$SERVER_BACKEND" in
    vllm)
       uv run python -m uvicorn server_vllm:app --host $SERVER_HOSTNAME --port 2244 --workers 1
        ;;
    normal|transformers|baseline)
        uv run python -m uvicorn server:app --host $SERVER_HOSTNAME --port 2244 --workers 1
        ;;
    *)
        echo "Unknown LLM server backend '$SERVER_BACKEND'. Use 'vllm' or 'normal'." >&2
        exit 2
        ;;
esac

