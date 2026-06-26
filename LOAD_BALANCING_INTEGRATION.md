# Load Balancing Integration

This document explains how the load balancing infrastructure is integrated with the vLLM server.

## Overview

The load balancing system allows multiple vLLM inference servers to be managed by a central load balancer that distributes requests using a least-connections algorithm. The integration is controlled by the `USE_LOAD_BALANCING` constant.

## Key Components

### 1. Configuration Constants

In `src/cfg/constants_titanic.py`:

```python
USE_VLLM = os.getenv("LLMGE_USE_VLLM", "1").lower() in ("1", "true", "yes")
USE_LOAD_BALANCING = os.getenv("LLMGE_USE_LOAD_BALANCING", "0").lower() in ("1", "true", "yes")

# Load balancing configuration
SERVER_REGISTRY_FILE = os.getenv("SERVER_REGISTRY_FILE", os.path.join(ROOT_DIR, "servers.json"))
LOAD_BALANCER_PORT = int(os.getenv("LOAD_BALANCER_PORT", "9000"))
LOADBALANCER_LOG_FILE = os.getenv("LOADBALANCER_LOG_FILE", os.path.join(ROOT_DIR, "loadbalancer.log"))
```

#### Configuration Validation

The constants module includes automatic validation to prevent misconfiguration. If `USE_VLLM=False` but vLLM-specific environment variables are set, an error will be raised:

```
ValueError: Configuration Error: vLLM-specific environment variables are set, but USE_VLLM=False.
The following vLLM-specific variables are configured:
  - TENSOR_PARALLEL_SIZE=2
  - VLLM_DTYPE=bfloat16

To fix this issue, either:
  1. Set LLMGE_USE_VLLM=1 to enable vLLM, or
  2. Unset the vLLM-specific environment variables
```

The following vLLM-specific variables are checked:
- `TENSOR_PARALLEL_SIZE`
- `GPU_MEMORY_UTILIZATION`
- `MAX_MODEL_LEN`
- `VLLM_DTYPE`
- `VLLM_QUANTIZATION`
- `ENABLE_PREFIX_CACHING`
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE`
- `VLLM_WORKER_MULTIPROC_METHOD`
- `VLLM_USE_FLASHINFER_SAMPLER`
- `VLLM_ATTENTION_BACKEND`

This validation prevents hard-to-debug issues where vLLM settings are accidentally configured but ignored.

### 2. Load Balancing Files

- **`load_balancer.py`**: Main load balancer server (FastAPI app)
  - Discovers servers from `servers.json` registry
  - Performs health checks
  - Routes requests to least-loaded server
  - Implements retry logic

- **`load_balancer.sh`**: Slurm script to launch the load balancer
  - Initializes the server registry
  - Starts the load balancer on port 9000 (default)

- **`start_cluster.sh`**: Convenience script to start a full cluster
  - Starts load balancer
  - Starts N inference servers
  - Handles registry initialization

- **`manage_servers.sh`**: Utilities for managing the cluster
  - `status`: Show server status
  - `cleanup`: Remove dead servers
  - `ports`: Show port usage

- **`monitor_cluster.sh`**: Monitoring script for cluster health

## How It Works

### When `USE_LOAD_BALANCING=True`

1. **Server Registry Setup**
   - The generated `server.sh` exports load balancing environment variables
   - Creates an empty `servers.json` registry if it doesn't exist
   - Each vLLM server instance registers itself in the shared registry file

2. **Server Registration**
   - When a vLLM server starts, it writes its hostname and port to `servers.json`
   - The load balancer periodically reads this file to discover available servers

3. **Load Balancer Discovery**
   - Load balancer polls `servers.json` every 10 seconds for new/removed servers
   - Performs health checks every 30 seconds on each registered server
   - Marks unhealthy servers after 3 consecutive failures

4. **Request Routing**
   - Client sends request to load balancer (e.g., `http://loadbalancer:9000/generate`)
   - Load balancer selects server with lowest load score (active connections + response time)
   - Forwards request to selected server
   - Implements retry logic (up to 2 retries) if server fails

### When `USE_LOAD_BALANCING=False` (Default)

- Server registry file is not created
- vLLM servers run independently on fixed ports
- Clients connect directly to individual servers
- No load balancing or automatic failover

## Usage

### Enable Load Balancing

Set the environment variable before running `slurm.py`:

```bash
export LLMGE_USE_LOAD_BALANCING=1
python slurm.py
```

Or set it in your `.env` file:

```bash
LLMGE_USE_LOAD_BALANCING=1
```

Then regenerate the server scripts:

```bash
python slurm.py
```

### Start a Load-Balanced Cluster

1. **Start the load balancer:**
   ```bash
   sbatch load_balancer.sh
   ```

2. **Wait for load balancer to be ready**, then start servers:
   ```bash
   sbatch server.sh
   sbatch server.sh
   sbatch server.sh
   ```

   Or use the convenience script:
   ```bash
   ./start_cluster.sh 3  # Start 3 servers
   ```

3. **Check cluster status:**
   ```bash
   ./manage_servers.sh status
   ```

4. **Monitor server pool:**
   ```bash
   LB_HOST=$(cat loadbalancer.log)
   curl http://$LB_HOST:9000/servers
   ```

### Client Configuration

When load balancing is enabled, update your client code to send requests to the load balancer instead of individual servers:

```python
import requests

# Without load balancing
# url = f"http://{server_hostname}:2244/generate"

# With load balancing
url = f"http://{loadbalancer_hostname}:9000/generate"

response = requests.post(url, json={
    "prompt": "Your prompt here",
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.8
})
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMGE_USE_LOAD_BALANCING` | `0` | Enable/disable load balancing |
| `SERVER_REGISTRY_FILE` | `{ROOT_DIR}/servers.json` | Path to server registry |
| `LOAD_BALANCER_PORT` | `9000` | Load balancer listening port |
| `LOADBALANCER_LOG_FILE` | `{ROOT_DIR}/loadbalancer.log` | Load balancer hostname file |

## Server Registry Format

The `servers.json` file has the following structure:

```json
{
  "servers": [
    {
      "hostname": "node123.cluster.edu",
      "port": 2244,
      "registered_at": "2026-06-15T10:30:00"
    },
    {
      "hostname": "node124.cluster.edu",
      "port": 2244,
      "registered_at": "2026-06-15T10:30:05"
    }
  ]
}
```

## Load Balancer API

### POST `/generate`
Route inference request to least-loaded server.

**Request:**
```json
{
  "prompt": "string",
  "max_new_tokens": 1024,
  "top_p": 0.8,
  "temperature": 0.7,
  "job_id": "string",
  "gene_id": "string"
}
```

**Response:**
```json
{
  "generated_text": "string",
  "response_time_sec": 1.23,
  "_latency_sec": 1.25,
  "evaluationScore": 0.95,
  ...
}
```

### GET `/servers`
Get status of all servers in the pool.

**Response:**
```json
{
  "total_servers": 3,
  "healthy_servers": 3,
  "total_active_requests": 5,
  "servers": [
    {
      "hostname": "node123",
      "port": 2244,
      "active_requests": 2,
      "total_requests": 150,
      "successful_requests": 148,
      "failed_requests": 2,
      "is_healthy": true,
      "load_score": 200.5
    }
  ]
}
```

### GET `/`
Health check endpoint.

## Architecture Diagram

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP requests
       ▼
┌──────────────────┐      reads      ┌──────────────┐
│  Load Balancer   │ ◄────────────── │ servers.json │
│  (port 9000)     │                 └──────────────┘
└────────┬─────────┘                        ▲
         │ forwards to                      │ registers
         │ least-loaded                     │ at startup
         ▼                                  │
    ┌─────────┐                             │
    │ vLLM    │─────────────────────────────┘
    │ Server 1│
    └─────────┘
    ┌─────────┐
    │ vLLM    │─────────────────────────────┘
    │ Server 2│
    └─────────┘
    ┌─────────┐
    │ vLLM    │─────────────────────────────┘
    │ Server 3│
    └─────────┘
```

## Benefits

1. **Load Distribution**: Automatically distributes requests across multiple servers
2. **Auto-discovery**: New servers are automatically discovered from registry
3. **Health Monitoring**: Unhealthy servers are automatically removed from rotation
4. **Retry Logic**: Failed requests are automatically retried on other servers
5. **Scalability**: Easy to add/remove servers without client configuration changes
6. **Metrics**: Tracks request counts, response times, and success rates per server

## Troubleshooting

### Load balancer can't find servers
- Check that `SERVER_REGISTRY_FILE` points to the same file for both load balancer and servers
- Verify servers successfully registered: `cat servers.json`
- Check load balancer logs: `tail -f slurm-{LB_JOB_ID}.out`

### Servers marked as unhealthy
- Check server health endpoint: `curl http://server:2244/`
- Review server logs for errors
- Verify network connectivity between load balancer and servers

### Cleanup dead servers
```bash
./manage_servers.sh cleanup
```

## Future Enhancements

- Add authentication/authorization
- Implement weighted load balancing
- Add request queueing with priority
- Support for sticky sessions (route same gene_id to same server)
- Prometheus metrics export
- WebSocket support for streaming responses
