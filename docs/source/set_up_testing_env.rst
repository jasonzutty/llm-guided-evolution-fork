Setting Up the Unit Testing Environment for the GitHub Actions Runner
=====================================================================

This tutorial explains how to prepare the GitHub Actions self-hosted runner on
PACE-ICE for unit testing.


Step 1: Start the Runner with tmux
----------------------------------

Use ``tmux`` so the runner keeps running even if you disconnect from SSH.

Create a new ``tmux`` session:

.. code-block:: bash

   tmux new -s runner

Inside the ``tmux`` session, go to the GitHub Actions runner directory:

.. code-block:: bash

   cd ~/actions-runner

Start the runner:

.. code-block:: bash

   ./run.sh

If the runner starts correctly, you should see output similar to:

.. code-block:: text

   Listening for Jobs

Detach from the ``tmux`` session without stopping the runner:

.. code-block:: text

   Ctrl + b, then press d

To reconnect to the runner session later:

.. code-block:: bash

   tmux attach -t runner

To list active ``tmux`` sessions:

.. code-block:: bash

   tmux ls


Step 2: Configure Kaggle Secrets on GitHub
------------------------------------------

The Titanic dataset download phase requires Kaggle credentials.

On the GitHub website:

1. Go to your repository.
2. Open ``Settings``.
3. Select ``Secrets and variables`` from the sidebar.
4. Click ``Actions``.
5. Click ``New repository secret``.
6. Create the following repository secrets:

   * ``KAGGLE_USERNAME``
   * ``KAGGLE_KEY``

The ``KAGGLE_USERNAME`` secret should contain your Kaggle username.

The ``KAGGLE_KEY`` secret should contain the API token from your Kaggle account.

To get your Kaggle API token:

1. Go to Kaggle.
2. Open your account settings.
3. Scroll to the API section.
4. Click ``Create New Token``.