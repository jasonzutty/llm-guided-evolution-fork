import os
import yaml

from src.cfg import constants



def replace_script_configuration(file_path, new_config):
    if not os.path.isabs(file_path):
        file_path = os.path.join(constants.ROOT_DIR, file_path)
    with open(file_path, 'w') as f:
        f.write(new_config if new_config.endswith('\n') else new_config + '\n')


def save_to_yaml(llm, python, gpu, islands, file_path=constants.SLURM_CONFIG_DIR):
    yaml_data = {
        "gpu_selection": gpu,
        "python_bash_script": python,
        "llm_bash_script": llm,
        "islands_bash_script": islands,
    }
    with open(f'{file_path}/slurm_config.yaml', 'w') as f:
        yaml.dump(yaml_data, f, indent=4)


def parse_config_sections(content):
    """Parse the configuration file into named sections."""
    sections = {}
    current_name = None
    current_lines = []
    in_section = False
    
    for line in content:
        stripped = line.strip()
        if stripped == "------":
            if in_section:
                # End of section content
                sections[current_name] = "\n".join(current_lines)
                current_lines = []
                in_section = False
                current_name = None
            else:
                # Start of section content
                in_section = True
        elif not in_section and stripped:
            # This is a section name
            current_name = stripped
        elif in_section:
            current_lines.append(line)
    
    return sections


if __name__ == "__main__":
    configuration_path = os.path.join(constants.SLURM_CONFIG_DIR, f"{constants.CLUSTER}.txt")

    with open(configuration_path, 'r') as file:
        content = [item.strip() for item in file.readlines()]

    sections = parse_config_sections(content)

    # Generate run.sh
    # Pass LLM_AVAIL flag to run_improved.py
    llm_avail_arg = "--llm_avail True" if constants.LLM_AVAIL else "--llm_avail False"
    run_sh = sections.get("run.sh", "") + f"""
echo "launching LLM Guided Evolution"
hostname
module load uv

export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)
uv run python run_improved.py {constants.OUTPUT_DIR} {llm_avail_arg}
"""
    replace_script_configuration("run.sh", run_sh)

    # Generate src/mixt.sh
    mixt_sh = sections.get("mixt.sh", "") + f"""
echo "Launching AIsurBL"
hostname
module load gcc/13.2.0
module load uv
source ~/.bashrc
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"
uv run python llm_crossover.py '{constants.SLURM_MIXT_INPUT_X}' '{constants.SLURM_MIXT_INPUT_Y}' '{constants.SLURM_MIXT_OUTPUT}'  --top_p {constants.SLURM_MIXT_TOP_P}   --temperature {constants.SLURM_MIXT_TEMPERATURE} --apply_quality_control '{constants.SLURM_MIXT_APPLY_QUALITY_CONTROL}' --bit {constants.SLURM_MIXT_BIT}
"""
    replace_script_configuration("src/mixt.sh", mixt_sh)

    # Get LLM GPU constraint
    llm_gpu = sections.get("llm-gpu", "").strip()

    # Generate python evaluation script template
    python_script = sections.get("python-bash-script", "") + f"""
echo "Launching Python Evaluation"
hostname

module load cuda
module load uv
export CUDA_VISIBLE_DEVICES=0
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

# Run Python script
{{}}
"""

    # Generate LLM bash script template
    # Check if LLM is available - if not, generate a simpler script
    if constants.LLM_AVAIL:
        llm_script = sections.get("llm-bash-script", "") + f"""
echo "Launching AIsurBL"
hostname

module load cuda
module load uv
export CUDA_VISIBLE_DEVICES=0
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

# Run Python script
{{}}
"""
    else:
        # LLM_AVAIL=False: Skip LLM server, just run basic Python
        llm_script = sections.get("llm-bash-script", "") + f"""
echo "LLM_AVAIL=False: Running in random mode (no LLM server needed)"
hostname

module load uv
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

# Run Python script (LLM operations will use random seed selection)
{{}}
"""

    # Generate islands bash script template (for islands_wrapper.py)
    # Pass LLM_AVAIL flag
    llm_avail_arg = "--llm_avail True" if constants.LLM_AVAIL else "--llm_avail False"
    islands_script = sections.get("islands", sections.get("island-controller", "")) + f"""
cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda

export HF_HOME={constants.HF_HOME}

# Run Python script
uv run python run_improved.py --checkpoints {{checkpoint_path}} --global_path {{global_path}} --llm_model {{llm_model}} --prompt_group {{prompt_group}} {llm_avail_arg}
"""

    # Generate server.sh
    if constants.LLM_AVAIL:
        # Normal server launch when LLM is available
        local_llm_server = f"""
echo "launching LLM Server"
# Optional chained submission count to work around walltime limits
COUNT=${{1:-1}}

# Backend selection:
# 1) Positional argument $2
# 2) Env var LLMGE_SERVER_BACKEND
# 3) Python constants.LLM_SERVER_BACKEND (derived from USE_VLLM)
SERVER_BACKEND=${{2:-${{LLMGE_SERVER_BACKEND:-{constants.LLM_SERVER_BACKEND}}}}}

# vLLM package (only used if SERVER_BACKEND=vllm)
VLLM_PACKAGE=${{VLLM_PACKAGE:-vllm==0.8.5}}

hostname
module load cuda
module load uv

# Respect Slurm's GPU assignment. If Slurm did not set CUDA_VISIBLE_DEVICES,
# fall back to the two local device ordinals requested by this job.
export CUDA_DEVICE_ORDER="${{CUDA_DEVICE_ORDER:-PCI_BUS_ID}}"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0,1}}"

# Cache dirs (used for both backends)
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="$UV_CACHE_DIR/xdg"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export FLASHINFER_CACHE_DIR="$XDG_CACHE_HOME/flashinfer"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_CACHE_DIR"

# Prevent transformers from importing TensorFlow (vLLM doesn't need it)
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3

echo "Using UV cache: $UV_CACHE_DIR"
echo "Using XDG cache: $XDG_CACHE_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# vLLM-specific env: only meaningful if SERVER_BACKEND=vllm
export NCCL_DEBUG="${{NCCL_DEBUG:-WARN}}"
export NCCL_IB_DISABLE="${{NCCL_IB_DISABLE:-1}}"
export NCCL_P2P_DISABLE="${{NCCL_P2P_DISABLE:-1}}"
export NCCL_SHM_DISABLE="${{NCCL_SHM_DISABLE:-0}}"
export VLLM_WORKER_MULTIPROC_METHOD="${{VLLM_WORKER_MULTIPROC_METHOD:-spawn}}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${{VLLM_DISABLE_CUSTOM_ALL_REDUCE:-true}}"
export TENSOR_PARALLEL_SIZE="${{TENSOR_PARALLEL_SIZE:-2}}"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "VLLM_DISABLE_CUSTOM_ALL_REDUCE=$VLLM_DISABLE_CUSTOM_ALL_REDUCE"

# FlashInfer tuning (again, only relevant if vLLM path is used)
export VLLM_USE_FLASHINFER_SAMPLER="${{VLLM_USE_FLASHINFER_SAMPLER:-0}}"
export VLLM_ATTENTION_BACKEND="${{VLLM_ATTENTION_BACKEND:-FLASH_ATTN}}"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "FLASHINFER_CACHE_DIR=$FLASHINFER_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)
HOSTNAME_FILE=$(pwd)"/hostname.log"
echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"

# Load balancing configuration
export LLMGE_USE_LOAD_BALANCING=${{LLMGE_USE_LOAD_BALANCING:-{str(constants.USE_LOAD_BALANCING).lower()}}}
export SERVER_REGISTRY_FILE=${{SERVER_REGISTRY_FILE:-{constants.SERVER_REGISTRY_FILE}}}
export LOAD_BALANCER_PORT=${{LOAD_BALANCER_PORT:-{constants.LOAD_BALANCER_PORT}}}

if [ "$LLMGE_USE_LOAD_BALANCING" = "true" ]; then
    echo "Load balancing ENABLED"
    echo "  Registry file: $SERVER_REGISTRY_FILE"
    echo "  Load balancer port: $LOAD_BALANCER_PORT"

    # Initialize registry file if it doesn't exist
    if [ ! -f "$SERVER_REGISTRY_FILE" ]; then
        echo "Creating empty server registry: $SERVER_REGISTRY_FILE"
        echo '{{"servers": []}}' > "$SERVER_REGISTRY_FILE"
    fi
else
    echo "Load balancing DISABLED"
fi

echo "Starting LLM server on host: $SERVER_HOSTNAME (count=$COUNT, backend=$SERVER_BACKEND)"
echo "Using vLLM package: $VLLM_PACKAGE"

# Log the island controller setting for debugging
echo "SUBMIT_ISLAND_CONTROLLER=${{SUBMIT_ISLAND_CONTROLLER:-<not set>}}"

# Default behavior: START island controller unless explicitly disabled
if [ "${{SUBMIT_ISLAND_CONTROLLER:-1}}" = "1" ]; then
    # Submit the paired island-controller job from here so the two stay in sync.
    echo "Submitting island controller (count=$COUNT)"
    sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"
else
    echo "Skipping island controller submission (SUBMIT_ISLAND_CONTROLLER=${{SUBMIT_ISLAND_CONTROLLER}})"
fi

case "$SERVER_BACKEND" in
    vllm)
       uv run python -m uvicorn server_vllm:app --host $SERVER_HOSTNAME --port {constants.PORT} --workers 1
        ;;
    normal|transformers|baseline)
        uv run python -m uvicorn server:app --host $SERVER_HOSTNAME --port {constants.PORT} --workers 1
        ;;
    *)
        echo "Unknown LLM server backend '$SERVER_BACKEND'. Use 'vllm' or 'normal'." >&2
        exit 2
        ;;
esac

"""
    else:
        # LLM_AVAIL=False: Run island controller directly without LLM server
        # The server.sh job still runs, but instead of starting LLM server,
        # it just keeps running and launches the island controller, then waits
        local_llm_server = f"""
echo "LLM_AVAIL=False: Running in random mode (no LLM server needed)"
echo "This job will coordinate island controller without starting LLM inference"

# Optional chained submission count to work around walltime limits
COUNT=${{1:-1}}

hostname
module load uv

export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
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
"""

    server_config = sections.get("server-sh", "")
    replace_script_configuration("server.sh", server_config + local_llm_server)
    print(f"Generated server.sh with config (LLM_AVAIL={constants.LLM_AVAIL}):\n{server_config}")

    # Generate unified island_controller.sbatch
    island_controller = sections.get("island-controller", sections.get("islands", "")) + f"""
# Island controller arguments:
# $1 (COUNT): Number of remaining restart iterations
# $2 (PREV_SERVER_JOB_ID): Job ID of the currently-running server (to cancel)
COUNT=$1
PREV_SERVER_JOB_ID=${{2:-}}

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda

# LLM_Storage
export HF_HOME={constants.HF_HOME}

# Change to the repository root
cd {constants.ROOT_DIR}

uv run python islands_wrapper.py {constants.ISLAND_CONTROLLER_RUN_NAME} \\
    --num_islands {constants.ISLAND_CONTROLLER_NUM_ISLANDS} \\
    --llms {constants.ISLAND_CONTROLLER_LLMS} \\
    --prompt_groups "{constants.ISLAND_CONTROLLER_PROMPT_GROUPS}"

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Controller completed; launching next server iteration with count: $NEXT_COUNT"
    sbatch server.sh "$NEXT_COUNT"
fi
"""
    # Remove problematic constraint lines that might cause "Invalid feature specification"
    # The -C intel constraint often fails on clusters
    island_controller = island_controller.replace("#SBATCH -C intel", "# Node constraint removed (was causing Invalid feature specification)")

    island_controller = island_controller.replace(
        "#SBATCH --job-name=Islands",
        "#SBATCH --job-name=IslandsController"
    )
    replace_script_configuration("island_controller.sbatch", island_controller)
    print(f"Generated island_controller.sbatch")

    # Save templates to YAML for runtime use
    save_to_yaml(llm_script, python_script, llm_gpu, islands_script)
    print(f"Saved configuration to {constants.SLURM_CONFIG_DIR}/slurm_config.yaml")
