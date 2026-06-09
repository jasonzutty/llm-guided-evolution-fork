#!/bin/bash
#SBATCH --job-name=Island_controller
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=run_job_outputs/islands/Report_island_controller-%j.out
#SBATCH -C intel
# Island controller arguments:
# $1 (COUNT): Number of remaining restart iterations
# $2 (PREV_SERVER_JOB_ID): Job ID of the currently-running server (to cancel)
COUNT=$1
PREV_SERVER_JOB_ID=${2:-}

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda

# LLM_Storage
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Change to the repository root
cd /storage/ice1/2/5/mgullapalli6/llm-guided-evolution-infrastructure

uv run python islands_wrapper.py mujoco_islands_run \
    --num_islands 1 \
    --llms llama3 \
    --prompt_groups "Mujoco/Normal"

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Controller completed; launching next server iteration with count: $NEXT_COUNT"
    sbatch server.sh "$NEXT_COUNT"
fi
