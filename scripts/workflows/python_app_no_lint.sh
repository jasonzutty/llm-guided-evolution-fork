#!/usr/bin/env bash
set -euo pipefail

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

prepare_cifar10() {
  local archive="cifar-10-python.tar.gz"
  local target_dir="sota/ExquisiteNetV2"

  if [ ! -f "$archive" ]; then
    curl -O "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  fi

  if [ ! -d "$target_dir/cifar-10-batches-py" ]; then
    tar -xzf "$archive" -C "$target_dir/"
  fi

  (
    cd "$target_dir"
    uv run split.py
  )
}

install_uv
uv sync
prepare_cifar10

# Load CUDA module if running on HPC with module system
if command -v module >/dev/null 2>&1; then
  module load cuda 2>/dev/null || echo "CUDA module not available"
fi

# Ensure CUDA is visible to PyTorch
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Disable LLM server auto-start for tests that don't need it (like ExquisiteNetV2)
# Use -v for verbose output and -s to disable output capture (show print statements)
export LLMGE_AUTO_START_SERVER=0
uv run pytest -v -s
