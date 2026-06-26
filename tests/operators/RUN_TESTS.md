# Running Evolution Loop Tests

## Problem

The `test_evo_loop.py::test_individual` test hangs because it waits for:
1. **LLM server** to be ready (up to 3600 seconds / 1 hour)
2. **SLURM job** to complete (up to 30 hours)

## Solution

Set these environment variables to bypass the waits:

```bash
# Quick test run (completes in ~7-8 seconds)
cd /home/mgullapalli6/scratch/llm-guided-evolution-infrastructure
LOCAL_LLM=false DELAYED_CHECK=true pytest tests/operators/test_evo_loop.py::test_individual -v
```

Or with uv:
```bash
LOCAL_LLM=false DELAYED_CHECK=true uv run pytest tests/operators/test_evo_loop.py::test_individual -v
```

## What the environment variables do:

- **`LOCAL_LLM=false`**: Skips waiting for the LLM server to become ready
  - Default: `true` (waits for `hostname.log` and HTTP server)
  - Set to `false` to bypass the wait
  
- **`DELAYED_CHECK=true`**: Returns immediately after submitting jobs, doesn't wait for completion
  - Default: `true` in SLURM mode, `false` in LOCAL mode
  - Explicitly setting it ensures the test doesn't hang waiting for jobs

## Alternative: Run all operator tests

```bash
LOCAL_LLM=false DELAYED_CHECK=true pytest tests/operators/ -v
```

## For CI/CD

Add to your test script:
```bash
export LOCAL_LLM=false
export DELAYED_CHECK=true
pytest tests/operators/
```
