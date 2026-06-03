# Analysis

Centralized location for analysis scripts and dashboard-related work for llm-guided-evolution.

## Purpose
This folder is for post-run analysis only. Scripts here should read experiment outputs, logs, and result files without modifying raw artifacts.

## Structure
- `scripts/`: runnable analysis scripts
- `utils/`: shared helper functions
- `dashboard/`: future dashboard-related code
- outputs should be written to `results/analysis/`

## Conventions
- do not modify raw run outputs
- scripts should run end-to-end
- prefer reusable helper functions over duplicated logic
- use descriptive snake_case filenames
