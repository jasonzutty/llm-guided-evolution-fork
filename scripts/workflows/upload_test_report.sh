#!/usr/bin/env bash
set -euo pipefail

GH_TOKEN="${GH_TOKEN:?GH_TOKEN is required}"
REPO="${REPO:?REPO is required, for example owner/repo}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_NUMBER="${RUN_NUMBER:-local}"
ACTOR="${ACTOR:-$(id -un 2>/dev/null || echo local)}"
HEAD_SHA="${HEAD_SHA:-$(git rev-parse HEAD)}"
SERVER_URL="${SERVER_URL:-https://github.com}"
API_URL="${API_URL:-https://api.github.com}"

download_artifact() {
  local artifact_name="$1"
  local output_zip="$2"
  local required="${3:-false}"

  local artifacts_json artifact_id
  artifacts_json=$(curl -fsSL \
    -H "Authorization: Bearer $GH_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "$API_URL/repos/$REPO/actions/runs/$RUN_ID/artifacts")

  artifact_id=$(printf '%s' "$artifacts_json" | ARTIFACT_NAME="$artifact_name" python3 -c '
import json
import os
import sys

data = json.load(sys.stdin)
for artifact in data.get("artifacts", []):
    if artifact.get("name") == os.environ["ARTIFACT_NAME"]:
        print(artifact["id"])
        break
')

  if [ -z "${artifact_id:-}" ]; then
    if [ "$required" = "true" ]; then
      echo "Artifact ${artifact_name} not found for run ${RUN_ID}"
      exit 1
    fi
    return 1
  fi

  echo "Found ${artifact_name} artifact id: $artifact_id"
  curl -fL \
    -H "Authorization: Bearer $GH_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "$API_URL/repos/$REPO/actions/artifacts/$artifact_id/zip" \
    -o "$output_zip"
}

mkdir -p downloaded-report

download_artifact "junit-report" "downloaded-report/junit-report.zip" true
unzip -o downloaded-report/junit-report.zip -d downloaded-report
ls -R downloaded-report
test -f downloaded-report/report.xml

if download_artifact "doc-suggestions" "downloaded-report/doc-suggestions.zip" false; then
  unzip -o downloaded-report/doc-suggestions.zip -d downloaded-report || true
else
  echo "[]" > downloaded-report/doc_suggestions.json
fi

UTC_TIME="$(date -u +'%Y-%m-%d_%H-%M-%S')"
DISPLAY_TIME="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
RUN_DIR="runs/${UTC_TIME}_run-${RUN_NUMBER}"
COMMIT_TITLE="$(git log -1 --pretty=%s "$HEAD_SHA")"
COMMIT_MESSAGE="$(git log -1 --pretty=%B "$HEAD_SHA")"
COMMIT_SHORT_SHA="$(git rev-parse --short "$HEAD_SHA")"

if [ ! -d gh-pages/.git ]; then
  rm -rf gh-pages
  mkdir gh-pages
  (
    cd gh-pages
    git init
    git checkout -b gh-pages
  )
fi

mkdir -p "gh-pages/${RUN_DIR}"
cp downloaded-report/report.xml "gh-pages/${RUN_DIR}/report.xml"

if [ -f downloaded-report/doc_suggestions.json ]; then
  cp downloaded-report/doc_suggestions.json "gh-pages/${RUN_DIR}/doc_suggestions.json"
else
  echo "[]" > "gh-pages/${RUN_DIR}/doc_suggestions.json"
fi

cat > "gh-pages/${RUN_DIR}/metadata.json" <<EOF
{
  "time": "${DISPLAY_TIME}",
  "user": "${ACTOR}",
  "run_id": "${RUN_ID}",
  "run_number": "${RUN_NUMBER}",
  "commit_sha": "${HEAD_SHA}",
  "commit_short_sha": "${COMMIT_SHORT_SHA}",
  "commit_title": $(COMMIT_TITLE="$COMMIT_TITLE" python3 -c 'import json,os; print(json.dumps(os.environ["COMMIT_TITLE"]))'),
  "commit_message": $(COMMIT_MESSAGE="$COMMIT_MESSAGE" python3 -c 'import json,os; print(json.dumps(os.environ["COMMIT_MESSAGE"]))'),
  "report_path": "${RUN_DIR}/report.xml",
  "actions_url": "${SERVER_URL}/${REPO}/actions/runs/${RUN_ID}",
  "commit_url": "${SERVER_URL}/${REPO}/commit/${HEAD_SHA}"
}
EOF

python3 <<'PY'
import json
from pathlib import Path

root = Path("gh-pages")
runs_dir = root / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)

(root / ".nojekyll").write_text("", encoding="utf-8")
runs_list = [path.name for path in sorted(runs_dir.glob("*"), reverse=True) if path.is_dir()]
(root / "runs" / "runs_list.json").write_text(json.dumps(runs_list), encoding="utf-8")
PY

(
  cd gh-pages
  git config user.name "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git add .
  git diff --cached --quiet && exit 0
  git commit -m "Add report for run ${RUN_NUMBER}"
)
