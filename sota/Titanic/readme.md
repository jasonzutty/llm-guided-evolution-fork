# README For VIP Bootcamp Students
## Working on PACE-ICE
LLM-GE is resource intensive when running with local LLMS. There are ways to configure LLM-GE to run with API calls, but for this readme, we are going to use local models.
You will want to begin by logging on to PACE-ICE. You can do this by following the instructions [Here](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042100).

Once logged into PACE-ICE, you will land on a login node. 
> [!IMPORTANT]
> You should not be doing any work on a login node, as soon as you land on one you should put yourself on a remote session on a compute node. 
> For this guide, we recommend requesting a GPU for this session so that you can ensure you set up a proper environment. You can do so with `srun -G 1 --pty bash`.

We recommend installing a tool called uv for Python package management. Instructions for the install can be found [Here](https://docs.astral.sh/uv/getting-started/installation/). But on this system, you can simply type `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Once uv is installed, you can set up your Python environment from within your LLM-GE directory by running `uv sync --cache-dir ~/scratch/.uv`.

> [!NOTE]
> When running on PACE-ICE you have a limited quote in your home directory, which is why we specify using the cache folder on the scratch partition, where your quota is much larger

## Preparing Titanic Problem
We don't want to check in data into our repository, so instead we check in scripts to pull the data. We have prepared a script to download the titanic dataset here, but you will have to first set up an API key with kaggle to utilize it. If you don't want to go through this, you can simply drop the train.csv file into sota/Titanic/data. Otherwise, please follow the instructions [Here](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials).

After following the instructions, from the sota/Titanic directory, run the command `uv run ./pull_data.sh`.

You can run the example preprocessing script with `uv run preprocess.py` to produce the data/processed_train.csv file expected by the model template script.

>[!NOTE]
> LLM-GE is set up to run code borrowed from the [Titanic Top Solution](https://www.kaggle.com/code/soham1024/titanic-data-science-eda-with-meme-solution) notebook. To get comparable results to your previous experiments, you will need to put your own pre-processing code into this sota/Titanic pipeline by replacing the preprocess.py to export your own data/processed_train.csv file.

When running LLM-GE, it will expect that:
- data will be in data/processed_train.csv with truth data
- a seed model will be in model.py and conform to the scikit api
- eval script will run, import the model, train and score on the processed_train.csv

You can test this by running `uv run eval.py` to get out a false positive and false negative score.

## Set Up LLM-GE
By default, LLM-GE is set up to run to evolve an image classifier for the CIFAR-10 dataset. We will need to change the configurations to work with the Titanic problem, as well as prepare our evolution to run on pace-ice.

Ensure that your settings are up to date in src/cfg/constants_titanic.py, and then constants.py is currently symlinked to it.

You will then prepare your scripts by running `uv run slurm.py`. This will generate bash files that you can submit to slurm

Next you will need to submit an inference server that the LLM-GE will use for mating and mutating individuals. You will do this by running: `sbatch server.sh`, this will submit the server script that slurm.py created.

Monitor the output and then when you are ready, you will run `sbatch run.sh` to kick off an evolution!

Since the `run.sh` script and the `server.sh` script only run for 8 hours each you will have to resubmit the jobs after they time out. You can automate this process using the `-d afterany:<job_id>` and `-d after:<job_id>` commands for sbatch. The `afterany` keyword will queue up a job to run after another job finishes. The `after` keyword will trigger a job to be submitted once another job starts. Since the server job is a resource intensive job it often has much longer time in the queue than the run job. To avoid having the run client active without a server use the `after` keyword to ensure a server is active once the client starts. You can use these command to build a dependancy chain that will execute one job after another until all jobs complete. Here is an example:

```
[amcdaniel39@atl1-1-02-003-19-1 llm-guided-evolution-fork]$ sbatch server.sh
Submitted batch job 3194630
[amcdaniel39@atl1-1-02-003-19-1 llm-guided-evolution-fork]$ sbatch -d after:3194630 run.sh
Submitted batch job 3194631
[amcdaniel39@atl1-1-02-003-19-1 llm-guided-evolution-fork]$ sbatch -d afterany:3194630 server.sh
Submitted batch job 3194632
[amcdaniel39@atl1-1-02-003-19-1 llm-guided-evolution-fork]$ sbatch -d after:3194632 run.sh
Submitted batch job 3194633
[amcdaniel39@atl1-1-02-003-19-1 llm-guided-evolution-fork]$ sbatch -d afterany:3194632 server.sh
Submitted batch job 3194634
[amcdaniel39@atl1-1-02-003-19-1 llm-guided-evolution-fork]$ sbatch -d after:3194634 run.sh
Submitted batch job 3194635
```

These commands will chain 3 server/client jobs one after another, effectively letting the server run for 24 hours! NOTE: running jobs at scale should only be done AFTER you confirm that it runs once. Otherwise it becomes increasingly difficult to debug, and potentially uses far more resources on pace than necessary making it difficult for others to get jobs running. Don't scale up your resources without careful verification that it makes sense to.
