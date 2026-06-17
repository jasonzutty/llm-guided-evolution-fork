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

uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude .venv
uv run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude .venv
uv run pytest -v
