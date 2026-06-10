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

uv run pytest
