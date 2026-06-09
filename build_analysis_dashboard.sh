#!/bin/bash
#SBATCH --job-name=analysis_dashboard
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=analysis_dashboard-%j.out
#SBATCH -C intel

# where analysis results are saved
output_dir=${1:-"analysis/results/analysis"}
# where run logs are saved
input_log_dir=${2:-"run_job_outputs"}

mkdir -p ${output_dir}

echo "summarizing slurm output"
uv run python analysis/scripts/summarize_slurm.py --input ${input_log_dir} --output ${output_dir}/slurm_summary.csv

echo "getting run inventory"
uv run python analysis/scripts/inventory_runs.py --input . --output ${output_dir}/run_inventory.csv

echo "extrating metrics"
uv run python analysis/scripts/extract_metrics.py --input . --output ${output_dir}/run_metrics.csv

echo "building entity summary"
uv run python analysis/scripts/entity_metrics_summary.py --input ${output_dir}/run_metrics.csv --output ${output_dir}/entity_metrics_summary.csv --report-output ${output_dir}/entity_metrics_report.md

echo "building failure report"
uv run python analysis/scripts/build_failure_report.py --input ${output_dir}/slurm_summary.csv --counts-output ${output_dir}/failure_counts.csv --report-output ${output_dir}/failure_report.md

echo "comparing runs"
uv run python analysis/scripts/compare_runs.py --slurm-input ${output_dir}/slurm_summary.csv --inventory-input ${output_dir}/run_inventory.csv --metrics-input ${output_dir}/run_metrics.csv --output ${output_dir}/run_comparison.csv

echo "summarizing slurm output"
uv run python analysis/scripts/export_dashboard_data.py --slurm-input ${output_dir}/slurm_summary.csv --comparison-input ${output_dir}/run_comparison.csv --failure-input ${output_dir}/failure_counts.csv --metrics-input ${output_dir}/run_metrics.csv --output-dir ${output_dir}

echo "summarizing slurm output"
uv run python analysis/scripts/plot_controller_activity.py --input ${output_dir}/slurm_summary.csv --output ${output_dir}/controller_activity_summary.png

echo "summarizing slurm output"
uv run python analysis/scripts/plot_entity_status.py --input ${output_dir}/entity_metrics_summary.csv --output ${output_dir}/entity_status_summary.png

echo "summarizing slurm output"
uv run python analysis/scripts/plot_fitness_quality.py --input ${output_dir}/entity_metrics_summary.csv --output ${output_dir}/fitness_quality_summary.png

echo "summarizing slurm output"
uv run python analysis/scripts/generate_leaderboard.py --input ${output_dir}/entity_metrics_summary.csv --output ${output_dir}/top_10_leaderboard.csv

echo "summarizing slurm output"
uv run python analysis/scripts/generation_timeline.py

echo "summarizing slurm output"
uv run python analysis/scripts/build_static_dashboard.py --input-dir ${output_dir} --output ${output_dir}/dashboard.html