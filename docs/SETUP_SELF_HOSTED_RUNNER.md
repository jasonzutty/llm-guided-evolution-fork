# Setting Up Self-Hosted GitHub Actions Runner on PACE HPC

## Step 1: Get Runner Token from GitHub

1. Go to your GitHub repository
2. Click **Settings** → **Actions** → **Runners**
3. Click **New self-hosted runner**
4. Select **Linux** as the operating system
5. GitHub will show you commands with a token - **keep this page open**

## Step 2: SSH into PACE and Set Up Runner

```bash
# SSH into PACE
ssh YOUR_USERNAME@login-ice.pace.gatech.edu

# Create directory for the runner
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download the runner (use the EXACT command GitHub shows you)
# It will look something like this:
curl -o actions-runner-linux-x64-2.319.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz

# Extract
tar xzf ./actions-runner-linux-x64-*.tar.gz

# Configure the runner (use the EXACT command GitHub shows with YOUR token)
# Add labels: self-hosted, linux, slurm
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO --token YOUR_TOKEN --labels self-hosted,linux,slurm,hpc

# Answer the prompts:
# - Runner group: Default
# - Runner name: pace-ice-runner (or whatever you want)
# - Work folder: _work (default)
```

## Step 3: Create Slurm Job to Keep Runner Active

Create a file `~/actions-runner/start-runner.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=github-runner
#SBATCH -t 168:00:00
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH --output=runner-%j.out
#SBATCH --error=runner-%j.err

cd ~/actions-runner

# Load necessary modules
module load cuda
module load uv

# Run the runner
./run.sh
```

## Step 4: Submit the Runner Job

```bash
cd ~/actions-runner
sbatch start-runner.sbatch

# Check if it's running
squeue -u $USER

# Check the output
tail -f runner-*.out
```

You should see: "Listening for Jobs"

## Step 5: Update GitHub Workflow

The workflow file `.github/workflows/python-app.yml` is already configured to use:

```yaml
runs-on: [self-hosted, linux, slurm]
```

This matches the labels we set during configuration!

## Step 6: Test It

Push a commit and watch the Actions tab - your job should run on your HPC runner!

## Maintenance

**To check runner status:**
```bash
squeue -u $USER
tail -f ~/actions-runner/runner-*.out
```

**To restart runner:**
```bash
# Cancel the old job
scancel JOB_ID

# Submit new one
cd ~/actions-runner
sbatch start-runner.sbatch
```

**When runner expires (after 7 days):**
Just resubmit the sbatch job. The runner will reconnect automatically.

## Troubleshooting

**Runner not appearing in GitHub:**
- Check the output: `tail -f ~/actions-runner/runner-*.out`
- Verify it says "Listening for Jobs"
- Check labels match: `self-hosted, linux, slurm`

**Jobs not picking up the runner:**
- Ensure the workflow has `runs-on: [self-hosted, linux, slurm]`
- Verify runner is "Idle" status in GitHub Settings → Actions → Runners

**Runner keeps stopping:**
- Check Slurm job status: `squeue -u $USER`
- Check if it hit walltime limit: `sacct -j JOB_ID`
- Resubmit with `sbatch start-runner.sbatch`

## Security Note

The runner has access to your HPC resources and GitHub secrets. Only use for repos you trust!
