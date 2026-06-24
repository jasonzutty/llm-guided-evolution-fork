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

Next you will need to submit an inference server that the LLM-GE will use for mating and mutating individuals. You will do this by running: `sbatch server.sh`, this will submit the server script that slurm.py created. This will also submit the main evolution process seperately through the `island_controller.sbatch` file.

First make sure that the process runs successfully. Then on subsequent runs you can use the command line argument to `server.sh` to make it resume once the 8 hour time limit is reached. For example the command `sbatch server.sh 3` will run the server 3 times in a row, totalling 24 hours of run time.

NOTE: running jobs at scale with large numbers of consecutive resubmissions should only be done AFTER you confirm that it runs once. Otherwise it becomes increasingly difficult to debug, and potentially uses far more resources on pace than necessary making it difficult for others to get jobs running. Don't scale up your resource utilization without careful verification that it makes sense to.
