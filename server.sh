#!/bin/bash
#SBATCH --job-name=LLMGE01_Server
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH -G 2

#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/slurm-%j.out
echo "LLM_AVAIL=False: Running in random mode (no LLM server needed)"
echo "This job will coordinate island controller without starting LLM inference"

# Optional chained submission count to work around walltime limits
COUNT=${1:-1}

hostname
module load uv

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

# Create a dummy hostname file so pipeline doesn't wait for it
export SERVER_HOSTNAME=$(hostname)
HOSTNAME_FILE=$(pwd)"/hostname.log"
echo "Writing dummy hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"

echo "Submitting island controller (count=$COUNT) in random mode"
sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"

# Keep this job alive to maintain coordination
# In random mode, this acts as a coordinator rather than a server
echo "Coordinator job running. Pipeline will use random seed model selection."
echo "Waiting for island controller to complete..."

# Sleep indefinitely to keep job alive while island controller runs
# The walltime limit will eventually terminate this
while true; do
    sleep 300  # Check every 5 minutes

    # Check if island controller is still running
    if [ -n "$SLURM_JOB_ID" ]; then
        # Check if any dependent jobs are still queued/running
        DEPENDENT_JOBS=$(squeue -u $USER -h -o "%A" -d "$SLURM_JOB_ID" 2>/dev/null | wc -l)
        if [ "$DEPENDENT_JOBS" -eq "0" ]; then
            echo "No dependent jobs remain. Coordinator exiting."
            break
        fi
    fi
done

echo "Coordinator job complete"
