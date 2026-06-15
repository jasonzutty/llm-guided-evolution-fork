# Summary: Load Balancing Integration with Configuration Validation

## What Was Done

### 1. Added Load Balancing Control (`USE_LOAD_BALANCING`)

**File**: `src/cfg/constants_titanic.py`

Added a new constant to enable/disable load balancing:
```python
USE_LOAD_BALANCING = os.getenv("LLMGE_USE_LOAD_BALANCING", "0").lower() in ("1", "true", "yes")
```

Also added related configuration constants:
- `SERVER_REGISTRY_FILE` - Path to the shared server registry JSON
- `LOAD_BALANCER_PORT` - Port for the load balancer (default: 9000)
- `LOADBALANCER_LOG_FILE` - File containing load balancer hostname

### 2. Added Configuration Validation for vLLM Settings

**File**: `src/cfg/constants_titanic.py`

Implemented validation to catch configuration errors early. If `USE_VLLM=False` but vLLM-specific environment variables are set, a clear error is raised:

```python
if not USE_VLLM:
    _vllm_specific_vars = {
        "TENSOR_PARALLEL_SIZE": os.getenv("TENSOR_PARALLEL_SIZE"),
        "GPU_MEMORY_UTILIZATION": os.getenv("GPU_MEMORY_UTILIZATION"),
        # ... other vLLM vars ...
    }
    
    _set_vllm_vars = {k: v for k, v in _vllm_specific_vars.items() if v is not None}
    
    if _set_vllm_vars:
        raise ValueError("Configuration Error: vLLM-specific environment variables are set...")
```

### 3. Updated Server Script Generation

**File**: `slurm.py`

Modified the `server.sh` generation to:
- Export `LLMGE_USE_LOAD_BALANCING` environment variable
- Export load balancing configuration paths
- Initialize `servers.json` registry if load balancing is enabled
- Print load balancing status at startup

Generated `server.sh` now includes:
```bash
# Load balancing configuration
export LLMGE_USE_LOAD_BALANCING=${LLMGE_USE_LOAD_BALANCING:-false}
export SERVER_REGISTRY_FILE=${SERVER_REGISTRY_FILE:-/path/to/servers.json}
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
```

### 4. Created Documentation

Created three documentation files:

1. **`LOAD_BALANCING_INTEGRATION.md`** - Complete technical documentation
   - Architecture overview
   - Component descriptions
   - Configuration validation details
   - API reference
   - Troubleshooting guide

2. **`LOAD_BALANCING_QUICKSTART.md`** - Quick setup guide
   - How to enable/disable load balancing
   - Starting a cluster
   - Monitoring commands
   - Testing instructions

3. **`test_config_validation.py`** - Automated validation tests
   - Tests valid configurations
   - Tests invalid configurations
   - Tests validation error messages

## How It Works

### With Load Balancing Enabled (`USE_LOAD_BALANCING=True`)

1. User sets environment variable: `export LLMGE_USE_LOAD_BALANCING=1`
2. Run `python slurm.py` to regenerate `server.sh` with load balancing support
3. Start load balancer: `sbatch load_balancer.sh`
4. Start vLLM servers: `sbatch server.sh` (multiple times)
5. Each server registers itself in `servers.json`
6. Load balancer discovers servers from registry and routes requests

### With Load Balancing Disabled (`USE_LOAD_BALANCING=False`, default)

1. Default behavior - no load balancing
2. No registry file is created
3. Servers run independently on fixed ports
4. Clients connect directly to individual servers

## Configuration Validation Benefits

The new validation catches errors like:

**Bad Configuration** ❌:
```bash
export LLMGE_USE_VLLM=0
export TENSOR_PARALLEL_SIZE=2
export VLLM_DTYPE=bfloat16
```

**Error Message**:
```
ValueError: Configuration Error: vLLM-specific environment variables are set, but USE_VLLM=False.
The following vLLM-specific variables are configured:
  - TENSOR_PARALLEL_SIZE=2
  - VLLM_DTYPE=bfloat16

To fix this issue, either:
  1. Set LLMGE_USE_VLLM=1 to enable vLLM, or
  2. Unset the vLLM-specific environment variables
```

This prevents:
- Silent configuration errors
- Hard-to-debug issues where settings are ignored
- Wasted time debugging why vLLM settings aren't taking effect

## Testing

Run the validation tests:
```bash
python test_config_validation.py
```

Expected output:
```
============================================================
Testing Configuration Validation
============================================================
Test 1: Valid config (USE_VLLM=True with vLLM settings)...
  ✓ PASSED: No error raised

Test 2: Invalid config (USE_VLLM=False with vLLM settings)...
  ✓ PASSED: Correct error raised

Test 3: Valid config (USE_VLLM=False without vLLM settings)...
  ✓ PASSED: No error raised

============================================================
Results: 3/3 tests passed
============================================================
```

## Usage Examples

### Enable Load Balancing

```bash
# Set environment variable
export LLMGE_USE_LOAD_BALANCING=1

# Regenerate server.sh
python slurm.py

# Start cluster
./start_cluster.sh 3  # Start load balancer + 3 servers
```

### Disable Load Balancing

```bash
# Unset or set to 0
export LLMGE_USE_LOAD_BALANCING=0

# Regenerate server.sh
python slurm.py

# Start individual servers
sbatch server.sh
```

### Check Configuration

```bash
# Verify load balancing status in generated server.sh
grep "LLMGE_USE_LOAD_BALANCING" server.sh

# Test for configuration errors
python -c "from src.cfg import constants"
```

## Files Modified

1. **`src/cfg/constants_titanic.py`**
   - Added `USE_LOAD_BALANCING` constant
   - Added load balancing configuration constants
   - Added vLLM configuration validation

2. **`slurm.py`**
   - Updated server.sh generation with load balancing support

## Files Created

1. **`LOAD_BALANCING_INTEGRATION.md`** - Full documentation
2. **`LOAD_BALANCING_QUICKSTART.md`** - Quick start guide
3. **`test_config_validation.py`** - Validation tests

## Existing Files (Unchanged)

These files work seamlessly with the integration:
- `load_balancer.py` - Load balancer server
- `load_balancer.sh` - Load balancer Slurm script
- `start_cluster.sh` - Cluster startup script
- `manage_servers.sh` - Cluster management utilities
- `monitor_cluster.sh` - Monitoring script
- `server_vllm.py` - vLLM server (no changes needed)

## Key Design Decisions

1. **Default to Disabled**: `USE_LOAD_BALANCING` defaults to `False` to maintain backward compatibility

2. **Environment Variable Control**: Can be overridden via `LLMGE_USE_LOAD_BALANCING` env var

3. **Early Validation**: Configuration errors are caught at import time, not runtime

4. **Clear Error Messages**: Validation errors provide actionable fix instructions

5. **No Server Modifications**: Server code (`server_vllm.py`) doesn't need changes - registration is handled by the shell script

## Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMGE_USE_VLLM` | `1` | Enable/disable vLLM backend |
| `LLMGE_USE_LOAD_BALANCING` | `0` | Enable/disable load balancing |
| `SERVER_REGISTRY_FILE` | `{ROOT_DIR}/servers.json` | Server registry path |
| `LOAD_BALANCER_PORT` | `9000` | Load balancer port |
| `LOADBALANCER_LOG_FILE` | `{ROOT_DIR}/loadbalancer.log` | Load balancer hostname file |

## Next Steps

To use the load balancing feature:

1. **Enable it**:
   ```bash
   export LLMGE_USE_LOAD_BALANCING=1
   python slurm.py
   ```

2. **Start a cluster**:
   ```bash
   ./start_cluster.sh 3
   ```

3. **Monitor the cluster**:
   ```bash
   ./manage_servers.sh status
   ```

4. **Send requests to load balancer** instead of individual servers:
   ```python
   # Old: direct to server
   # url = f"http://{server_host}:2244/generate"
   
   # New: via load balancer
   url = f"http://{loadbalancer_host}:9000/generate"
   ```

See `LOAD_BALANCING_QUICKSTART.md` for detailed usage instructions.
