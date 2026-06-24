Connecting PACE-ICE to GitHub Actions
=====================================

This tutorial explains how to connect a GitHub Actions self-hosted runner to
PACE-ICE so that CI/CD workflows can run on Georgia Tech's HPC infrastructure.

The goal is to register a Linux self-hosted runner on PACE-ICE and connect it
to a GitHub repository.

Step 1: Create a runner
--------------------------------------------

First, generate a self-hosted runner registration command from GitHub.

1. Navigate to your GitHub repository.
2. Go to:

   ``Settings > Actions > Runners``

3. Click:

   ``New self-hosted runner``

4. Select:

   * Operating system: ``Linux``
   * Architecture: ``x64``

5. GitHub will show a configuration command similar to the following:

   .. code-block:: bash

      ./config.sh --url https://github.com/Softicles/llm-guided-evolution-cicd --token <RUNNER_REGISTRATION_TOKEN>


Step 2: SSH into PACE-ICE
-------------------------

Connect to PACE-ICE using SSH.

.. code-block:: bash

   ssh <gtusername>@login-ice.pace.gatech.edu

Replace ``<gtusername>`` with your Georgia Tech username.

For example:

.. code-block:: bash

   ssh tnguyen831@login-ice.pace.gatech.edu

Step 3: Create a Runner Directory
---------------------------------

Create a dedicated directory for the GitHub Actions runner at the same level as the folder containing LLMGE (for example, saving both inside scratch).

.. code-block:: bash

   mkdir -p actions-runner
   cd actions-runner

This keeps the runner files separate from your project files.

Step 4: Download and configure the GitHub Actions Runner
------------------------------------------

For this step, come back to GitHub Action Runner set up guide and follow the instructions there to download the runner to the ``actions-runner`` folder.

During setup, GitHub may ask for:

* Runner name
* Runner group
* Runner labels
* Work directory

You can usually press ``Enter`` to accept the default values.


Step 5: Start the Runner
------------------------

After finish setting up, the directory should contain files such as:

* ``config.sh``
* ``run.sh``
* ``svc.sh``

Start the runner using:

.. code-block:: bash

   ./run.sh

After running the runner, go to your GitHub repository and open:

``Settings > Actions > Runners``

If the runner was configured correctly, it should be in "Idle" status, which means it is connected and waiting for jobs.

Inside your PACE-ICE terminal, you should see output similar to:

.. code-block:: text

   Listening for Jobs

.. important::

   If the runner shows a disk quota error during an automatic runner update, such as:

   .. code-block:: text

      An error occurred: Disk quota exceeded: '/home/.../actions-runner/_work/_update'

   From inside the ``actions-runner`` directory, clean the runner-generated files:

   .. code-block:: bash

      # remove diagnostic logs
      rm -rf _diag/*

      # remove job workspace, often huge
      rm -rf _work/*

      # optional: remove any cached temp files
      rm -rf _temp/* 2>/dev/null || true

   Then start the runner again

