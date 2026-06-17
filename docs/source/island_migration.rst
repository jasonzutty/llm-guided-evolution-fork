Island Migration
================

Overview
--------

Island Migration is a wrapper for the LLM-Guided Evolution (LLM-GE) framework, designed to enhance the performance of evolution using island-based migration strategies. It creates multiple islands, each of which runs LLM-GE with its own LLM and population of individuals, and handles the migration of individuals between these islands.

The codebase is structured to allow for easy integration of different LLMs and configurations, enabling users to experiment with various evolutionary strategies and LLMs.

At the same time, any original functionality of LLM-GE is preserved, allowing users to run the framework as they would normally.

Getting Started
---------------

To get started with Island Migration, follow these steps:

1. **Clone the Repository**: Clone the Island Migration repository from GitHub.

    ``git clone https://github.com/jasonzutty/llm-guided-evolution-fork``

2. **Install uv**: Instructions for installing ``uv`` can be found in the `uv documentation <https://github.com/astral-sh/uv>`_.

3. **Initialize the virtual environment**: Simply run ``uv venv``.

    ``uv venv``

4. **Sync Dependencies**: Sync the necessary dependencies using the ``uv sync`` command.

    ``uv sync``

5. **Symlink configs**: Symlink the right config file to constants.

    ``ln -sf constants_island_migration.py src/cfg/constants.py``

6. **Run the pipeline**, run slurm to set shell scripts and then run the pipeline.
    ``uv run slurm.py``
    ``sbatch server.sh -N`` (number of eras: generations * 5)



.. toctree::
   :maxdepth: 2
   :caption: Contents:

