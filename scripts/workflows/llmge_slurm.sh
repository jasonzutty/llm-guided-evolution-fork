#!/usr/bin/env bash
set -euo pipefail

SERVER_JOB_NAME="${SERVER_JOB_NAME:-LLMGE01_Server}"
SERVER_PORT="${SERVER_PORT:-2244}"
SLURM_TEST_SCRIPT="${SLURM_TEST_SCRIPT:-slurm-submit-test.sh}"
SERVER_SCRIPT="${SERVER_SCRIPT:-server.sh}"
REUSED_RUNNING_SERVER="false"

check_slurm_connectivity() {
  echo "HOSTNAME=$(hostname)"
  echo "SLURM_CONF=${SLURM_CONF:-<unset>}"
  which sbatch
  scontrol ping
  sinfo >/dev/null
}

prepare_runner_paths() {
  mkdir -p ~/scratch/.venv_runner
  ln -sfn ~/scratch/.venv_runner .venv
  ls -ld .venv
  readlink -f .venv

  mkdir -p ~/scratch/.cache_runner
  ln -sfn ~/scratch/.cache_runner .cache
  ls -ld .cache
  readlink -f .cache
}

prepare_titanic_data() {
  (
    cd sota/Titanic

    if [ -f "data/train.csv" ] && [ -f "data/processed_train.csv" ]; then
      echo "Titanic data already prepared."
      return
    fi

    if [ -z "${KAGGLE_KEY:-}" ]; then
      echo "ERROR: Titanic data is missing and KAGGLE_KEY is not set." >&2
      echo "Set KAGGLE_USERNAME and KAGGLE_KEY, or prepare sota/Titanic/data locally first." >&2
      exit 1
    fi

    export KAGGLE_API_TOKEN="$KAGGLE_KEY"
    uv run kaggle competitions list -s titanic

    if [ ! -f "data/train.csv" ]; then
      chmod +x pull_data.sh
      uv run ./pull_data.sh
    fi

    if [ ! -f "data/processed_train.csv" ]; then
      uv run python preprocess.py
    fi
  )
}

connectivity_check() {
  echo "Waiting for hostname.log..."

  for i in {1..120}; do
    if [ -s hostname.log ]; then
      SERVER_HOSTNAME=$(cat hostname.log)
      echo "Server hostname: $SERVER_HOSTNAME"
      break
    fi

    echo "hostname.log not ready yet... ($i/120)"
    sleep 10
  done

  if [ ! -s hostname.log ]; then
    echo "ERROR: hostname.log was not created"
    sacct -j "$SERVER_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
    return 1
  fi

  SERVER_HOSTNAME=$(cat hostname.log)
  SERVER_SHORT_HOSTNAME="${SERVER_HOSTNAME%%.*}"
  SERVER_URL=""

  echo "Waiting for LLM server on port ${SERVER_PORT}"
  echo "Host candidates: ${SERVER_HOSTNAME} ${SERVER_SHORT_HOSTNAME}"

  for i in {1..120}; do
    for candidate in "$SERVER_HOSTNAME" "$SERVER_SHORT_HOSTNAME"; do
      candidate_url="http://${candidate}:${SERVER_PORT}/"
      if curl -fsS --connect-timeout 5 --max-time 10 "$candidate_url" >/dev/null; then
        SERVER_URL="$candidate_url"
        echo "LLM server is ready at $SERVER_URL"
        break 2
      fi
    done

    echo "LLM server not ready yet... ($i/120)"
    sleep 10
  done

  if [ -z "$SERVER_URL" ] || ! curl -fsS --connect-timeout 5 --max-time 10 "$SERVER_URL" >/dev/null; then
    echo "ERROR: LLM server failed readiness check"
    sacct -j "$SERVER_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
    return 1
  fi

  echo "LLM server verified successfully"
  cp hostname.log .cache/hostname.log
  echo "Saved hostname.log to .cache/hostname.log"
}

start_llm_server() {
  echo "Checking for existing Slurm server job named ${SERVER_JOB_NAME}..."

  SERVER_JOB_ID=$(
    squeue -h \
      -u "$USER" \
      -n "$SERVER_JOB_NAME" \
      -t RUNNING \
      -o "%A" | head -n 1 || true
  )

  if [ -n "$SERVER_JOB_ID" ]; then
    echo "Found existing running server job: $SERVER_JOB_ID"
    echo "Checking for hostname.log..."

    if [ -s .cache/hostname.log ]; then
      cp .cache/hostname.log hostname.log
      echo "Copied .cache/hostname.log to hostname.log"
    elif [ -s hostname.log ]; then
      echo "Using existing hostname.log"
    else
      echo "ERROR: hostname.log not found for running server job."
      exit 1
    fi

    if [ -s hostname.log ]; then
      if ! connectivity_check; then
        echo "Connectivity check failed for existing running job; proceeding to fallback server startup path."
        SERVER_JOB_ID=""
      else
        REUSED_RUNNING_SERVER="true"
      fi
    fi
  fi

  if [ -z "$SERVER_JOB_ID" ]; then
    SERVER_JOB_ID=$(
      squeue -h \
        -u "$USER" \
        -n "$SERVER_JOB_NAME" \
        -t PENDING,CONFIGURING \
        -o "%A" | head -n 1 || true
    )

    if [ -n "$SERVER_JOB_ID" ]; then
      echo "Found existing queued/starting server job: $SERVER_JOB_ID"
      echo "Will wait for this job instead of submitting ${SERVER_SCRIPT} again."
    else
      echo "No existing ${SERVER_JOB_NAME} job found."
      echo "Submitting new LLM server job..."
      rm -f hostname.log
      SERVER_JOB_ID=$(sbatch --parsable "$SERVER_SCRIPT")
      echo "Submitted server job: $SERVER_JOB_ID"
    fi
  fi

  if [ "$REUSED_RUNNING_SERVER" != "true" ]; then
    connectivity_check
  fi
}

submit_test_job() {
  JOB_ID=$(sbatch --parsable "$SLURM_TEST_SCRIPT")
  echo "Submitted Slurm job: $JOB_ID"
}

monitor_test_job() {
  while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
    echo "Job $JOB_ID still running. Waiting 30 seconds..."
    sleep 30
  done

  pace-job-summary "$JOB_ID" || true
  echo "Job $JOB_ID finished."
}

verify_test_job_success() {
  JOB_STATE=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
  echo "State: $JOB_STATE"

  if [[ "$JOB_STATE" == "COMPLETED" ]]; then
    echo "Job $JOB_ID completed successfully."
  else
    echo "Job $JOB_ID failed with state: $JOB_STATE"
    exit 1
  fi
}

print_report_sections() {
  python3 - <<'PY'
import os
import sys
import xml.etree.ElementTree as ET

report = "tests/results/report.xml"
if not os.path.exists(report):
    print("No report found.")
    sys.exit(0)

root = ET.parse(report).getroot()

passed = [
    tc.get("classname", "") + "::" + tc.get("name", "")
    for tc in root.iter("testcase")
    if tc.find("failure") is None and tc.find("error") is None and tc.find("skipped") is None
]
print(f"Total passed: {len(passed)}\n")
for name in passed:
    print(f"  - {name}")

skipped = [
    (
        tc.get("classname", "") + "::" + tc.get("name", ""),
        tc.find("skipped").get("message", "No reason given"),
    )
    for tc in root.iter("testcase")
    if tc.find("skipped") is not None
]
print(f"\nTotal skipped: {len(skipped)}\n")
for name, reason in skipped:
    print(f"  - {name}")
    print(f"    Reason: {reason}")

failed = []
for tc in root.iter("testcase"):
    name = tc.get("classname", "") + "::" + tc.get("name", "")
    failure = tc.find("failure")
    error = tc.find("error")
    if failure is not None or error is not None:
        node = failure if failure is not None else error
        failed.append((name, node.get("message", ""), (node.text or "").strip()))

print(f"\nTotal failed: {len(failed)}\n")
for name, message, detail in failed:
    print(f"::group::FAILED: {name.split('::')[-1]}")
    print(f"  Test    : {name}")
    if message:
        print(f"  Message : {message}")
    if detail:
        print("  Detail  :")
        for line in detail.splitlines():
            print(f"    {line}")
    print("::endgroup::")

mutation_benchmarks = []
crossover_benchmarks = []
for tc in root.iter("testcase"):
    name = tc.get("name", "")
    lower_name = name.lower()
    if "benchmark" not in lower_name:
        continue
    entry_id = name.split("[")[-1].rstrip("]") if "[" in name else name
    stdout = tc.find("system-out")
    entry = {
        "id": entry_id,
        "skipped": tc.find("skipped"),
        "output": (stdout.text or "").strip() if stdout is not None else "",
    }
    if "mutation" in lower_name:
        mutation_benchmarks.append(entry)
    if "crossover" in lower_name:
        crossover_benchmarks.append(entry)

print(f"\nMutation benchmarks:  {len(mutation_benchmarks)}")
print(f"Crossover benchmarks: {len(crossover_benchmarks)}")
PY

  python3 src/doc_checker.py \
    "${BASE_SHA:-$(git rev-parse HEAD~1 2>/dev/null || git rev-parse HEAD)}" \
    "${HEAD_SHA:-$(git rev-parse HEAD)}"
}

check_slurm_connectivity
prepare_runner_paths
prepare_titanic_data
start_llm_server
submit_test_job
monitor_test_job
verify_test_job_success
print_report_sections
