import os
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
        


def unpackIslands(num_islands, checkpoints) -> list[Island]:
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
    for i in range(num_islands):
        curr_llm = ISLAND_LLMS[i]
        print("Unpacking island " + curr_llm, flush=True)
        checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)
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

def migrateIslands(topology, num_islands, checkpoints, gen):
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
    islands = unpackIslands(num_islands, checkpoints)
    
    print("MIGRATING INDIVIDUALS")
    migrate(topology, islands)

    print("PACKING ISLANDS")
    print()
    packIslands(islands, gen)



def submit_mutate_prompts(llm_model, n=5):
    """
    Submits a bash script to mutate prompts using the specified LLM model.
    
    Parameters
    ----------
    llm_model : str
        The LLM model to use for mutation.
    n : int
        The number of templates to mutate.

    Returns
    -------
    list prompt_job_ids
        A list of job IDs for the submitted mutation jobs.
    """
    prompt_job_ids = []
    templates = np.random.choice(glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt'), n)
    file_path = './mutate_prompts_temp.sh'
    for i, template in enumerate(templates):
        python_runline = f"python src/llm_prompt_mutation.py --llm_model {llm_model} --template {template}"
        script = LLM_BASH_SCRIPT_TEMPLATE.format(LLM_GPU, python_runline)

        with open(file_path, 'w') as file:
            file.write(script)
        print(f"\t‣ Bash Script Saved to {file_path}")

        job_id = None
        successful_sub_flag = False
        result = subprocess.run([RUN_COMMAND, file_path], capture_output=True, text=True)
        if result.returncode == 0:
            print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
            successful_sub_flag = True
            job_id = result.stdout.split('job ')[-1].strip()
        else:
            print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
            successful_sub_flag = False
            job_id = None
        
        prompt_job_ids.append(job_id)
    return prompt_job_ids
    

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
        gen = extract_generation(fname)
        max_gen = max(max_gen, gen)

    return max_gen if max_gen > 0 else 1

# Island Controller Script to handle creating islands and migrating individuals between them.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Generation')
    # Add arguments
    parser.add_argument('checkpoints', type=str, help='Save Dir')
    parser.add_argument('--num_islands', type=int, help='Number of Islands', default=2)
    # Parse the arguments
    args = parser.parse_args()
    island_script = "src/island_temp_script.sh"
    checkpoints = args.checkpoints
    num_islands = args.num_islands

    if num_islands > MAX_ISLANDS:
        print("Number of islands exceeds maximum allowed: " + str(MAX_ISLANDS))
        exit(1)


    # initialize the graph topology
    islands_list = []
    for i in range(num_islands):
        curr_llm = ISLAND_LLMS[i]
        checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)
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
        for i in range(num_islands):
            curr_llm = ISLAND_LLMS[i]
            print("Generating Island " + curr_llm, flush=True)
            checkpoint_path = os.path.join(checkpoints, "island_" + curr_llm)
            
            job_id = submit_run(island_script, ISLANDS_BASH_SCRIPT_TEMPLATE.format(curr_llm, checkpoint_path, global_path, curr_llm))
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

        '''
        # mutate prompts
        print("Mutating Prompts")
        prompt_job_ids = submit_mutate_prompts(LLM_MIXTRAL)
        done = True
        for i in range(len(prompt_job_ids)):
            done = check4job_completion(prompt_job_ids[i])
            if not done:
                break

        if not done:
            print("Error occured in loop, job not done")
            break
        '''

        # migrate individuals between islands
        if migration_gen != 0:
            print("Starting island migration on era " + str(era), flush=True)
            migrateIslands(topology, num_islands, checkpoints, era * migration_gen)

        print("Finished era " + str(era), flush=True)
    
    print("Finished evolutionary loop")