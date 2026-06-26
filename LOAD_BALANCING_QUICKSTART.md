# Quick Start: Using Load Balancing

## Enable Load Balancing

### Option 1: Environment Variable
```bash
export LLMGE_USE_LOAD_BALANCING=1
python slurm.py  # Regenerate server.sh with load balancing enabled
```

### Option 2: Set in .env file
Add to your `.env`:
```bash
LLMGE_USE_LOAD_BALANCING=1
```

Then regenerate:
```bash
python slurm.py
```

## Start the Cluster

### Method 1: Use start_cluster.sh (Recommended)
```bash
./start_cluster.sh 3  # Start load balancer + 3 servers
```

### Method 2: Manual Setup
```bash
# 1. Start load balancer
sbatch load_balancer.sh

# 2. Wait for it to start (check for loadbalancer.log file)
ls -l loadbalancer.log

# 3. Start servers (they will auto-register)
sbatch server.sh
sbatch server.sh
sbatch server.sh
```

## Monitor the Cluster

```bash
# Check cluster status
./manage_servers.sh status

# View server pool
LB_HOST=$(cat loadbalancer.log)
curl http://$LB_HOST:9000/servers | python -m json.tool

# Monitor load balancer logs
tail -f slurm-*.out | grep LoadBalancer
```

## Test the Setup

```bash
LB_HOST=$(cat loadbalancer.log)

# Test request
curl -X POST http://$LB_HOST:9000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def fibonacci(n):",
    "max_new_tokens": 100,
    "temperature": 0.7
  }'
```

## Disable Load Balancing

```bash
export LLMGE_USE_LOAD_BALANCING=0
python slurm.py  # Regenerate server.sh without load balancing
```

## Key Files

- `servers.json` - Server registry (auto-created)
- `loadbalancer.log` - Load balancer hostname
- `LOAD_BALANCING_INTEGRATION.md` - Full documentation

## Troubleshooting

**Issue**: Servers not appearing in registry
```bash
# Check registry file
cat servers.json

# Verify environment variable is set
grep LLMGE_USE_LOAD_BALANCING server.sh
```

**Issue**: Load balancer can't reach servers
```bash
# Test server health directly
SERVER_HOST=$(head -1 hostname.log)
curl http://$SERVER_HOST:2244/

# Check firewall/network rules
```

**Issue**: Dead servers in registry
```bash
# Clean up unhealthy servers
./manage_servers.sh cleanup
```
