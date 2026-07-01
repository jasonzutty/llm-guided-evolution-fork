# Analysis Pipeline

This is the standard post-run analysis flow for LLM-GE Titanic runs. Run these commands from the repository root:

```bash
cd to your root directory of the LLM-GE project.
```
## Quick Batch Version

```bash
mkdir -p analysis/results/analysis

Go to root directory.

run
sbatch build_analysis_dashboard.sh
```
Then run the generated HTML file using a HTML file viewer such as VSCode's Live server extension.


# More detailed documentation
##  Confirm Run Outputs Exist

Check that the run produced Slurm logs and experiment artifacts actually exist:

```bash
find run_job_outputs -name "*.out" | head
find . -name "global_gen_*.pkl" | head
```

The log analysis can run with only `run_job_outputs/`. Pareto/global-data analysis needs at least one `global_gen_*.pkl` file.

##  Prepare Analysis Output Directory

```bash
mkdir -p analysis/results/analysis
```

All generated analysis files should go under `analysis/results/analysis/`.

##  Summarize Slurm Logs

```bash
uv run python analysis/scripts/summarize_slurm.py \
  --input run_job_outputs \
  --output analysis/results/analysis/slurm_summary.csv
```

This creates a table of Slurm jobs, statuses, errors, and controller/server activity.

## Inventory Run Artifacts

```bash
uv run python analysis/scripts/inventory_runs.py \
  --input . \
  --output analysis/results/analysis/run_inventory.csv
```

This records available logs, result files, pickle files, and other run artifacts.

##  Extract Metrics

```bash
uv run python analysis/scripts/extract_metrics.py \
  --input . \
  --output analysis/results/analysis/run_metrics.csv
```

This scans logs and pickle artifacts for generation, fitness, entity, and run-level metrics.

##  Build Entity Summary

```bash
uv run python analysis/scripts/entity_metrics_summary.py \
  --input analysis/results/analysis/run_metrics.csv \
  --output analysis/results/analysis/entity_metrics_summary.csv \
  --report-output analysis/results/analysis/entity_metrics_report.md
```

This converts raw extracted metrics into a cleaner entity-level table and Markdown report.

##  Build Failure Report

```bash
uv run python analysis/scripts/build_failure_report.py \
  --input analysis/results/analysis/slurm_summary.csv \
  --counts-output analysis/results/analysis/failure_counts.csv \
  --report-output analysis/results/analysis/failure_report.md
```

Use this to quickly see common failure modes from Slurm logs.

##  Compare Runs

```bash
uv run python analysis/scripts/compare_runs.py \
  --slurm-input analysis/results/analysis/slurm_summary.csv \
  --inventory-input analysis/results/analysis/run_inventory.csv \
  --metrics-input analysis/results/analysis/run_metrics.csv \
  --output analysis/results/analysis/run_comparison.csv
```

This joins the Slurm, inventory, and metrics outputs into a run-level comparison table. NOTE: this does not compare many runs.

##  Export Dashboard Tables

```bash
uv run python analysis/scripts/export_dashboard_data.py \
  --slurm-input analysis/results/analysis/slurm_summary.csv \
  --comparison-input analysis/results/analysis/run_comparison.csv \
  --failure-input analysis/results/analysis/failure_counts.csv \
  --metrics-input analysis/results/analysis/run_metrics.csv \
  --output-dir analysis/results/analysis
```

This produces dashboard-ready CSV files such as `dashboard_runs.csv`, `dashboard_metrics.csv`, and `dashboard_failures.csv`.

##  Generate Charts And Leaderboard

```bash
uv run python analysis/scripts/plot_controller_activity.py \
  --input analysis/results/analysis/slurm_summary.csv \
  --output analysis/results/analysis/controller_activity_summary.png

uv run python analysis/scripts/plot_entity_status.py \
  --input analysis/results/analysis/entity_metrics_summary.csv \
  --output analysis/results/analysis/entity_status_summary.png

uv run python analysis/scripts/plot_fitness_quality.py \
  --input analysis/results/analysis/entity_metrics_summary.csv \
  --output analysis/results/analysis/fitness_quality_summary.png

uv run python analysis/scripts/generate_leaderboard.py \
  --input analysis/results/analysis/entity_metrics_summary.csv \
  --output analysis/results/analysis/top_10_leaderboard.csv
```

These create visual summaries and the top-10 entity leaderboard.

##  Generate Timeline Report

```bash
uv run python analysis/scripts/generation_timeline.py
```

This reads the standard files in `analysis/results/analysis/` and writes:

- `analysis/results/analysis/generation_timeline_report.md`
- `analysis/results/analysis/generation_timeline_graphs/`

##  Optional: Pareto Report From Global Data

If you have a generation pickle, point this command at it:

```bash
GLOBAL_PKL=$(find . -name "global_gen_*.pkl" | sort -V | tail -1)

uv run python analysis/scripts/global_data_fitness_report.py \
  --input "$GLOBAL_PKL" \
  --output analysis/results/analysis/global_data_pareto_report.md

uv run python analysis/scripts/pareto_summary_card.py
```

This produces:

- `analysis/results/analysis/global_data_pareto_report.md`
- `analysis/results/analysis/pareto_kpi_summary.md`
- `analysis/results/analysis/pareto_kpi_graphs/pareto_kpi_bar.png`

##  Optional: Per-Generation Fitness Diversity

If your run directory contains `global_gen_*.pkl` files, run:

```bash
uv run python fitness_diversity_quantifier.py <RUN_DIR> \
  --include-hist \
  --csv-name fitness_diversity_summary.csv
```

Replace `<RUN_DIR>` with the run directory that contains `global_data/global_gen_*.pkl`. The CSV is written inside `<RUN_DIR>`.

##  Optional: Per-Generation Pareto Images

```bash
uv run python create_pareto_graphs_by_generation.py <RUN_DIR> \
  --output-dir analysis/results/analysis/pareto_by_generation
```

Replace `<RUN_DIR>` with the directory that contains `global_gen_*.pkl` files. Images and GIFs are written to the chosen output directory.

##  Build Static Dashboard

Recommended on the cluster, because it does not require a web server, SSH tunnel, or Streamlit websocket connection:

```bash
uv run python analysis/scripts/build_static_dashboard.py \
  --input-dir analysis/results/analysis \
  --output analysis/results/analysis/dashboard.html
```

Open the generated file:

```text
analysis/results/analysis/dashboard.html
```
using the live server extension on VSCode (recommended).

or any other way to view HTML files in the browser (such as launching through Firefox).


This static page reads the CSVs, Markdown reports, and PNG charts generated by the previous steps.

##  Optional: Launch Streamlit Dashboard

Use Streamlit only if you have a working browser tunnel to the node running the app:

```bash
uv run python -m streamlit run analysis/dashboard/app.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.enableWebsocketCompression false \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
```

If Streamlit runs on a compute node, tunnel through a login node from your local machine:

```bash
ssh -N -L 8501:<compute-node-hostname>:8501 <username>@login-phoenix.pace.gatech.edu
```

Then open this locally:

```text
http://localhost:8501
```

The Streamlit app reads existing files from `analysis/results/analysis/` and can also analyze uploaded `.out`, `.log`, and `.pkl` files through the sidebar.

## Analysis Launcher Commands

These are the two dashboard launch/build commands you will usually want after running the analysis pipeline:

```bash
# Static dashboard, recommended
uv run python analysis/scripts/build_static_dashboard.py \
  --input-dir analysis/results/analysis \
  --output analysis/results/analysis/dashboard.html

# Streamlit dashboard, optional
uv run python -m streamlit run analysis/dashboard/app.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.enableWebsocketCompression false \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
```


