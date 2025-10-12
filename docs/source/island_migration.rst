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

    ``git clone https://github.com/heuryn/llm-guided-evolution-Island-Migration.git``

2. **Install uv**: Instructions for installing ``uv`` can be found in the `uv documentation <https://github.com/astral-sh/uv>`_.

3. **Initialize the virtual environment**: Simply run ``uv venv``.

    ``uv venv``

4. **Sync Dependencies**: Sync the necessary dependencies using the ``uv sync`` command.

    ``uv sync``

5. **Configure the Environment and run scripts**: Set up your environment variables and configurations as needed. The files that will need to be configured are:

    ``src/cfg/constants.py``

    ``src/cfg/pace_ice_scripts.py``
    
    ``src/cfg/icehammer_scripts.py``

    ``icehammer_island_controller.sbatch``

    ``pace_ice_island_controller.sbatch``

7. **Run the Evolution**: Execute the controller script to start the evolutionary process with island migration.

    ``uv run sbatch <cluster>_island_controller.sbatch``



.. toctree::
   :maxdepth: 2
   :caption: Contents:

