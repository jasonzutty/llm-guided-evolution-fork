#!/usr/bin/env python3
"""Test configuration validation for vLLM settings."""

import os
import sys
import subprocess

def test_valid_config():
    """Test that valid configs don't raise errors."""
    print("Test 1: Valid config (USE_VLLM=True with vLLM settings)...")

    env = os.environ.copy()
    env["LLMGE_USE_VLLM"] = "1"
    env["TENSOR_PARALLEL_SIZE"] = "2"
    env["VLLM_DTYPE"] = "bfloat16"

    result = subprocess.run(
        [sys.executable, "-c", "from src.cfg import constants"],
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("  ✓ PASSED: No error raised")
        return True
    else:
        print(f"  ✗ FAILED: Unexpected error:\n{result.stderr}")
        return False


def test_invalid_config():
    """Test that invalid configs raise errors."""
    print("\nTest 2: Invalid config (USE_VLLM=False with vLLM settings)...")

    env = os.environ.copy()
    env["LLMGE_USE_VLLM"] = "0"
    env["TENSOR_PARALLEL_SIZE"] = "2"
    env["VLLM_DTYPE"] = "bfloat16"
    env["ENABLE_PREFIX_CACHING"] = "true"

    result = subprocess.run(
        [sys.executable, "-c", "from src.cfg import constants"],
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0 and "vLLM-specific environment variables" in result.stderr:
        print(f"  ✓ PASSED: Correct error raised")
        print(f"\n  Error message preview:\n  {result.stderr.split(chr(10))[0]}...")
        return True
    elif result.returncode == 0:
        print("  ✗ FAILED: No error raised (should have raised ValueError)")
        return False
    else:
        print(f"  ✗ FAILED: Wrong error:\n{result.stderr}")
        return False


def test_vllm_disabled_no_settings():
    """Test that USE_VLLM=False without vLLM settings is fine."""
    print("\nTest 3: Valid config (USE_VLLM=False without vLLM settings)...")

    env = os.environ.copy()
    env["LLMGE_USE_VLLM"] = "0"
    # Remove any vLLM-specific vars
    for key in ["TENSOR_PARALLEL_SIZE", "VLLM_DTYPE", "ENABLE_PREFIX_CACHING",
                "GPU_MEMORY_UTILIZATION", "MAX_MODEL_LEN", "VLLM_QUANTIZATION"]:
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, "-c", "from src.cfg import constants"],
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("  ✓ PASSED: No error raised")
        return True
    else:
        print(f"  ✗ FAILED: Unexpected error:\n{result.stderr}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Testing Configuration Validation")
    print("="*60)

    results = []
    results.append(test_valid_config())
    results.append(test_invalid_config())
    results.append(test_vllm_disabled_no_settings())

    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)

    sys.exit(0 if all(results) else 1)
