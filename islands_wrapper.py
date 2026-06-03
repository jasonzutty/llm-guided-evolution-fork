import os
import re
import argparse
import glob
from deap import base, creator, tools
from deap.tools import HallOfFame
import subprocess
import time
import networkx as nx
from run_improved import load_checkpoint, save_checkpoint, extract_generation
from islands import Individual, Island, Topology, migrate, generate_graph_topology
from src.cfg.constants import *
from src.utils.print_utils import box_print

def submit_run(tempFile, text):
    """
    Bash script submission function for running a job on the cluster.
    Used for running each LLM's island.

    Parameters
    ----------
    tempFile : str
        Path to the temporary file where the bash script will be saved.
    text : str
        The content of the bash script to be executed.

    Returns
    -------
    job_id : str
        The job ID returned by the cluster after submitting the script.
        If submission fails, returns None.
    """
    with open(tempFile, 'w') as file:
        file.write(text)
    print(f"\t‣ Bash Script Saved to {tempFile}")
    job_id = None
    successful_sub_flag = False
    result = subprocess.run([RUN_COMMAND, tempFile], capture_output=True, text=True)
    if result.returncode == 0:
        print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
        successful_sub_flag = True
        job_id = result.stdout.split('job ')[-1].strip()
    else:
        print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
        successful_sub_flag = False
        job_id = None
    
    return job_id
    
def check_contents_for_error(contents):
    """
    Checks the output of a job for any signs of error.

    Parameters:
    contents (str): output of job to check for error

    Returns:
    bool: True if job completed successfully, False if error, None if neither.  
    """

    # Check for error indicators in the file
    if "traceback" in contents.lower() or "slurmstepd: error" in contents.lower():
        print("\t☠ Error Found in LLM Job Output.", flush=True)
        return False
    elif "end of era" in contents.lower():
        print("\t☑ LLM Job Completed Successfully.", flush=True)
        return True
    else:
        return None

def check4job_completion(job_id, local_output=None, check_interval=60, timeout=3600*30):
    """
    Check for the completion of a job by searching for its output file and scanning for errors.

    Parameters:
    job_id (str): The job ID to check.
    check_interval (int): Time in seconds between checks.
    timeout (int): Maximum time in seconds to wait for job completion.

    Returns:
    bool: True if job completed successfully, False otherwise.
    """

    if not job_id:
        print("Checking for job completion: job_id is None")
        return None

    if local_output is not None:
        state = check_contents_for_error(local_output)
        if state is None:
            raise Exception('Unexpected output from job')
        else:
            return state
    

    start_time = time.time()
    output_file = f'{SLURM_OUTPUT_PATH}islands/Report_islands-{job_id}.out'

    while True:
        # Check if the timeout is reached
        if time.time() - start_time > timeout:
            print("Timeout reached while waiting for job completion.")
            return False

        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                contents = file.read()
                state = check_contents_for_error(contents)
                if state is None:
                    pass
                else:
                    return state

        # Wait for some time before checking again
        time.sleep(check_interval)
        print(f'\t‣ Waiting on check4job_completion LLM job: {job_id} Time: {round(time.time() - start_time)}s Path: {output_file}', flush=True)
        


def unpackIslands(island_specs, checkpoints) -> list[Island]:
    """
    Unpacks islands from the checkpoints directory.

    Parameters
    ----------
    num_islands : int
        The number of islands to unpack.
    checkpoints : str
        The path to the checkpoints directory.

    Returns
    -------
    list[Island]
        A list of unpacked islands.
    """
    islands = []
    global_path = os.path.join(checkpoints, "global_data")
    for llm_name, prompt_group in island_specs:
        prompt_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", prompt_group)
        print(f"Unpacking island {llm_name} ({prompt_group})", flush=True)
        checkpoint_path = os.path.join(checkpoints, f"island_{llm_name}_{prompt_slug}")
        global_path = os.path.join(checkpoints, GLOBAL_DATA_PATH)
        checkpoint, start_gen, global_data = load_checkpoint(folder_name=checkpoint_path, global_path=global_path)
        
        if checkpoint:
            population = checkpoint["population"]
            hof = checkpoint["hof"]
        else:
            print("Missing Island ", checkpoint_path)
            exit(0)
        
        individuals = []
        for ind in population:
            individual = Individual(ind[0], ind.fitness.values)
            individuals.append(individual)
        
        island = Island(checkpoint_path, individuals)
        islands.append(island)
    return islands

def packIslands(islands: list[Island], gen: int):
    """
    Packs islands into their respective directories after migration.

    Parameters
    ----------
    islands : list[Island]
        A list of islands to pack.
    gen : int
        The current generation number.

    Returns
    -------
    None
    """
    for island in islands:
        island_path = island.path
        print("Packing island path" + island_path, flush=True)
        population = []
        for individual in island.individuals:
            ind = creator.Individual([individual.name])
            ind.fitness.values = individual.rank
            population.append(ind)
        
        hof = tools.HallOfFame(hof_size)
        checkpoint_data = {
            "population": population,
            "hof": hof,
        }
        save_checkpoint(gen=gen, folder_name=island_path, global_path=None, checkpoint_data=checkpoint_data)

def migrateIslands(topology, island_specs, checkpoints, gen):
    """
    Migrates individuals between islands based on the specified topology.

    Parameters
    ----------
    topology : Topology
        The graph topology defining the migration pattern.
    num_islands : int
        The number of islands involved in the migration.
    checkpoints : str
        The path to the checkpoints directory.
    gen : int
        The current generation number.

    Returns
    -------
    None
    """
    print("UNPACKING ISLANDS")
    islands = unpackIslands(island_specs, checkpoints)
    
    print("MIGRATING INDIVIDUALS")
    migrate(topology, islands)

    print("PACKING ISLANDS")
    print()
    packIslands(islands, gen)


def get_generation(global_path):
    """
    Scans the directory at global_path for checkpoint .pkl files and returns
    the highest generation number found.

    Parameters
    ----------
    global_path : str
        The path to the directory containing checkpoint files.

    Returns
    -------
    int
        The highest generation number found, or 1 if none found.
    """
    if not os.path.isdir(global_path):
        return 1

    max_gen = 0
    for fname in os.listdir(global_path):
        gen = extract_generation(fname, "global")
        if gen is not None:
            max_gen = max(max_gen, gen)

    return max_gen if max_gen > 0 else 1

# Island Controller Script to handle creating islands and migrating individuals between them.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Generation')
    # Add arguments
    parser.add_argument('checkpoints', type=str, help='Save Dir')
    parser.add_argument('--num_islands', type=int, help='Number of Islands', default=None)
    parser.add_argument('--llms', type=str, help='Comma-separated LLMs for islands (overrides ISLAND_LLMS)', default=None)
    parser.add_argument('--prompt_groups', type=str, help='Comma-separated prompt groups/globs aligned to islands', default=None)
    parser.add_argument('--prompt_group', type=str, help='Deprecated: single prompt group to apply to all islands', default=None)
    # Parse the arguments
    args = parser.parse_args()
    island_script = "src/island_temp_script.sh"
    checkpoints = args.checkpoints

    def parse_csv(arg_val):
        if not arg_val:
            return []
        return [item.strip() for item in arg_val.split(',') if item.strip()]

    cli_llms = parse_csv(args.llms)
    cli_prompts = parse_csv(args.prompt_groups)

    # Backwards compatibility: --prompt_group applies to all unless prompt_groups provided
    if args.prompt_group and not cli_prompts:
        cli_prompts = [args.prompt_group.strip()]

    base_llms = cli_llms if cli_llms else ISLAND_LLMS
    base_prompts = cli_prompts if cli_prompts else [DEFAULT_PROMPT_GROUP]

    if len(base_llms) == 0:
        print("No LLMs provided and ISLAND_LLMS is empty; cannot launch islands.")
        exit(1)

    # Build island specs: align lengths, or compute cartesian product when both lists have length >1 and differ
    island_specs = []
    if len(base_llms) == len(base_prompts):
        island_specs = list(zip(base_llms, base_prompts))
    elif len(base_llms) == 1 and len(base_prompts) > 1:
        island_specs = [(base_llms[0], pg) for pg in base_prompts]
    elif len(base_prompts) == 1 and len(base_llms) > 1:
        island_specs = [(llm, base_prompts[0]) for llm in base_llms]
    else:
        # different sizes >1: cartesian product
        for llm in base_llms:
            for pg in base_prompts:
                island_specs.append((llm, pg))

    num_islands = args.num_islands if args.num_islands is not None else len(island_specs)
    if num_islands != len(island_specs):
        print(f"num_islands ({num_islands}) does not match derived island specs ({len(island_specs)}). Adjust args or omit num_islands.")
        exit(1)

    if not cli_llms and num_islands > MAX_ISLANDS:
        print("Number of islands exceeds maximum allowed: " + str(MAX_ISLANDS))
        exit(1)

    # initialize the graph topology
    islands_list = []
    for llm_name, prompt_group in island_specs:
        prompt_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", prompt_group)
        checkpoint_path = os.path.join(checkpoints, f"island_{llm_name}_{prompt_slug}")
        island = Island(checkpoint_path, [])
        islands_list.append(island) 
    topology = generate_graph_topology(islands_list, Topology.FULL)

    global_path = os.path.join(checkpoints, GLOBAL_DATA_PATH)
    # start generation
    curr_gen = get_generation(global_path)
    start_era = curr_gen // migration_gen + 1 if migration_gen != 0 else 1
    for era in range(start_era, 2 if migration_gen == 0 else start_era + num_generations // migration_gen):
        print("Starting era " + str(era), flush=True)
        job_ids = []

        # submit island generation jobs
        for llm_name, prompt_group in island_specs:
            prompt_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", prompt_group)
            print(f"Generating Island {llm_name} with prompts {prompt_group}", flush=True)
            checkpoint_path = os.path.join(checkpoints, f"island_{llm_name}_{prompt_slug}")
            job_id = submit_run(island_script, ISLANDS_BASH_SCRIPT_TEMPLATE.format(checkpoint_path, global_path, llm_name, prompt_group))
            job_ids.append(job_id)
        
        # check island generation jobs for completion
        done = True
        for i in range(len(job_ids)):
            done = check4job_completion(job_ids[i])
            if not done:
                break
        
        if not done:
            print("Error occured in loop, job not done")
            break

        # migrate individuals between islands
        if migration_gen != 0:
            print("Starting island migration on era " + str(era), flush=True)
            migrateIslands(topology, island_specs, checkpoints, era * migration_gen)

        print("Finished era " + str(era), flush=True)
    
    print("Finished evolutionary loop")