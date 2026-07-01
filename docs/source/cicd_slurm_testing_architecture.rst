CI/CD Slurm Testing Architecture
================================

Purpose
-------

This workflow runs the project's tests on the ICE/PACE Slurm cluster through a self-hosted GitHub Actions runner.

It prepares the Python environment and Titanic data, starts the LLM server as a separate Slurm job, submits the test job, monitors completion, verifies success, summarizes test results, and uploads reports/artifacts.

Generic Architecture
--------------------

.. code-block:: text

   GitHub push / manual run
           |
           v
   GitHub Actions workflow
           |
           v
   Self-hosted runner on ICE
           |
           v
   Environment + data setup
           |
           v
   LLM server Slurm job: server.sh
           |
           v
   Server readiness check: hostname.log + curl
           |
           v
   Test Slurm job: slurm-submit-test.sh
           |
           v
   JUnit XML report + benchmark output
           |
           v
   GitHub artifact upload + documentation suggestions

Main Components
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Purpose
   * - GitHub Actions workflow
     - Orchestrates the full CI pipeline.
   * - Self-hosted runner
     - Connects GitHub Actions to the ICE/PACE cluster.
   * - Slurm connectivity check
     - Verifies that ``sbatch``, ``scontrol``, and ``sinfo`` work before running jobs.
   * - Scratch ``.venv`` symlink
     - Stores the Python environment in ``scratch`` to avoid home directory quota issues.
   * - Titanic data setup
     - Downloads and preprocesses data needed by the tests.
   * - ``server.sh``
     - Starts the LLM server as a separate Slurm job.
   * - ``hostname.log``
     - Passes the server node hostname back to the workflow.
   * - ``slurm-submit-test.sh``
     - Runs the actual test job on Slurm resources.
   * - ``tests/results/report.xml``
     - Stores JUnit-style test results.
   * - ``doc_checker.py``
     - Generates documentation suggestions from code changes.

System Interactions
-------------------

The workflow interacts with three external systems:

GitHub Actions
   Triggers the pipeline, runs orchestration logic, prints summaries, and uploads artifacts.

ICE/PACE Slurm Cluster
   Schedules the LLM server job and the test job on compute resources.

Kaggle API
   Downloads the Titanic dataset when ``train.csv`` is missing.

External References
-------------------

* `GitHub Actions documentation <https://docs.github.com/en/actions/get-started/understand-github-actions>`__
* `Self-hosted runners <https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners>`__
* `GitHub Actions workflow syntax <https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions>`__
* `GitHub Actions artifacts <https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts>`__
* `Slurm Workload Manager documentation <https://slurm.schedmd.com/documentation.html>`__
* `sbatch documentation <https://slurm.schedmd.com/sbatch.html>`__
* `squeue documentation <https://slurm.schedmd.com/squeue.html>`__
* `sacct documentation <https://slurm.schedmd.com/sacct.html>`__
* `Kaggle API documentation <https://github.com/Kaggle/kaggle-api>`__
* `pytest documentation <https://docs.pytest.org/>`__
* `uv documentation <https://docs.astral.sh/uv/>`__

Summary
-------

At a high level, GitHub Actions acts as the controller, the self-hosted runner acts as the bridge to ICE/PACE, Slurm runs the server and tests, and the workflow collects the final test and report artifacts back into GitHub.