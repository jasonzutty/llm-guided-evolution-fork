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

  # Check if file exists and is valid, remove if corrupted
  if [ -f "$archive" ]; then
    if ! gzip -t "$archive" 2>/dev/null; then
      echo "Existing file is corrupted, removing and re-downloading..."
      rm -f "$archive"
    fi
  fi

  # Download if file doesn't exist
  if [ ! -f "$archive" ]; then
    echo "Downloading CIFAR-10 dataset..."
    curl -fsSL -o "$archive" "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    # Verify the download is a valid gzip file
    if ! gzip -t "$archive" 2>/dev/null; then
      echo "Error: Downloaded file is not a valid gzip archive"
      rm -f "$archive"
      exit 1
    fi
    echo "Download successful and verified"
  fi

  if [ ! -d "$target_dir/cifar-10-batches-py" ]; then
    echo "Extracting CIFAR-10 dataset..."
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

# Disable LLM server auto-start for tests - conftest.py manages the server
export LLMGE_AUTO_START_SERVER=0

uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude .venv
uv run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude .venv
uv run pytest -v
