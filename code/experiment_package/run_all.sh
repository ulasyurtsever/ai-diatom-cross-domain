#!/usr/bin/env bash
# One-command driver for the IEEE Access R1 revision experiments.
# Runs every step, logs each to results_revision/logs/, and never aborts the
# whole run if a single step fails (so you get maximum output in one pass).
#
# Usage (from the released repo root that contains train.py + data/):
#   bash Experiment_Package/run_all.sh
# or from anywhere:
#   REPO_ROOT=/path/to/GitHub_Repo_Ready bash /path/to/Experiment_Package/run_all.sh

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
# fall back: if train.py isn't one level up, assume current dir is the repo
if [ ! -f "$REPO_ROOT/train.py" ]; then export REPO_ROOT="$(pwd)"; fi
PY="${PYTHON:-python3}"
LOG="$REPO_ROOT/results_revision/logs"; mkdir -p "$LOG"
echo "REPO_ROOT = $REPO_ROOT"
echo "Logs      = $LOG"

step () {  # step <name> <script>
  echo ""; echo "==================== $1 ===================="
  ( cd "$REPO_ROOT" && $PY "$HERE/$2" ) 2>&1 | tee "$LOG/$1.log"
  echo "---- $1 exit=${PIPESTATUS[0]} ----"
}

step 00_environment   capture_env.py
step 01_efficiency    exp4_inference_metrics.py      # fast, no training
step 02_multiseed     exp1_multiseed.py
step 03_dg_baselines  exp2_dg_baselines.py
step 04_ablation      exp3_ablation.py
step 05_sensitivity   exp5_sensitivity.py
step 06_aggregate     aggregate_results.py
step 07_posthoc       exp6_posthoc_analysis.py    # per-genus / confusion / failure (no training)
step 08_figures       make_revision_figures.py
step 09_export        export_all.py               # one archive with everything (no *.pth)

echo ""; echo "ALL DONE. See: $REPO_ROOT/results_revision/aggregated/REVISION_RESULTS_SUMMARY.md"
