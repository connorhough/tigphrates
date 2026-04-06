#!/usr/bin/env bash
#
# autoresearch loop runner
# Automates the run/eval/log/git cycle. Calls Claude only to decide
# what to change in train.py next.
#
# Usage:
#   ./python/run_experiments.sh [--dry-run] [TAG] [MAX_RUNS]
#
# Requires: python3, npx, git, claude (Claude Code CLI)
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DRY_RUN=false
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done

TAG="${POSITIONAL[0]:-}"
if [[ -z "$TAG" ]]; then
  TAG=$(date +%b%d | tr '[:upper:]' '[:lower:]')
fi
MAX_RUNS="${POSITIONAL[1]:-999}"
BRANCH="autoresearch/${TAG}"
TSV="results.tsv"
LOG="run.log"
TIMEOUT=720  # 12 min kill timeout

if $DRY_RUN; then
  export TIME_BUDGET=5
  export EVAL_GAMES=2
  TIMEOUT=30
  echo "*** DRY RUN: TIME_BUDGET=${TIME_BUDGET}s, EVAL_GAMES=${EVAL_GAMES}, TIMEOUT=${TIMEOUT}s ***"
fi

# macOS-compatible timeout wrapper (coreutils `timeout` not available by default)
run_with_timeout() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
  local watcher=$!
  wait "$pid" 2>/dev/null
  local rc=$?
  kill "$watcher" 2>/dev/null
  wait "$watcher" 2>/dev/null
  return $rc
}

# --- Setup ---
if ! git rev-parse --verify "$BRANCH" &>/dev/null; then
  git checkout -b "$BRANCH"
else
  git checkout "$BRANCH"
fi

if [[ ! -f "$TSV" ]]; then
  printf 'commit\twin_rate\tavg_min_score\tstatus\tdescription\n' > "$TSV"
fi

best_win_rate="0.000000"

# --- Helper: run one experiment ---
run_experiment() {
  echo ">>> Running training (5 min budget)..."
  run_with_timeout "$TIMEOUT" python3 python/train.py > "$LOG" 2>&1
  return $?
}

# --- Helper: extract results ---
extract_results() {
  local wr=$(grep "^win_rate:" "$LOG" 2>/dev/null | awk '{print $2}' || echo "")
  local ms=$(grep "^avg_min_score:" "$LOG" 2>/dev/null | awk '{print $2}' || echo "")
  echo "${wr:-CRASH} ${ms:-0.0}"
}

# --- Helper: log to TSV ---
log_result() {
  local commit="$1" wr="$2" ms="$3" status="$4" desc="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$commit" "$wr" "$ms" "$status" "$desc" >> "$TSV"
}

# --- Helper: discard last experiment (revert only train.py, not whole tree) ---
discard_last_experiment() {
  git checkout HEAD~1 -- python/train.py
  git reset HEAD~1 --soft
  git checkout HEAD -- python/train.py
}

# --- Helper: ask Claude for next experiment ---
ask_claude_for_edit() {
  local results_context best_wr
  results_context=$(cat "$TSV")
  best_wr="$best_win_rate"

  # Minimal, stateless prompt — only what Claude needs to decide
  local prompt
  prompt="You are an autonomous RL researcher. Modify python/train.py with ONE experimental idea.

RULES:
- Edit ONLY python/train.py. No other files.
- ONE idea per experiment. Keep changes focused.
- If win_rate=0 for baseline, prioritize reward shaping and larger rollouts.
- Deps: torch, numpy, gymnasium only. No new installs.
- Code must run: python python/train.py (5min budget, then 50-game eval).
- Be terse. No explanation needed. Just make the edit.

RESULTS SO FAR:
${results_context}

Best win_rate: ${best_wr}

Current train.py is at python/train.py — read it, then edit with your idea."

  claude --print --model claude-sonnet-4-6 --allowedTools Edit,Read,Bash,Grep,Glob -p "$prompt"
}

# --- Main loop ---
echo "=== autoresearch: ${BRANCH} ==="
echo "=== Max runs: ${MAX_RUNS} ==="
echo ""

for ((run=1; run<=MAX_RUNS; run++)); do
  echo "============================================"
  echo "  EXPERIMENT ${run}"
  echo "============================================"

  if [[ "$run" -eq 1 ]]; then
    echo ">>> Baseline run (no changes)"
  else
    echo ">>> Asking Claude for next experiment..."
    ask_claude_for_edit
  fi

  # Commit current state
  desc=$(git diff --stat python/train.py 2>/dev/null | tail -1 || echo "no changes")
  git add python/train.py
  if git diff --cached --quiet; then
    if [[ "$run" -eq 1 ]]; then
      desc="baseline PPO"
      git commit --allow-empty -m "experiment ${run}: ${desc}"
    else
      echo ">>> No changes made, skipping"
      continue
    fi
  else
    # Get a 1-line description from the diff
    desc=$(git diff --cached python/train.py | head -30 | grep "^+" | grep -v "^+++" | head -5 | tr '\n' ' ' | cut -c1-80)
    git commit -m "experiment ${run}: ${desc}"
  fi
  commit=$(git rev-parse --short HEAD)

  # Run experiment
  if run_experiment; then
    read -r wr ms <<< "$(extract_results)"

    if [[ "$wr" == "CRASH" ]]; then
      echo ">>> CRASH (no results in log)"
      tail -20 "$LOG"
      log_result "$commit" "0.000000" "0.0" "crash" "crashed: ${desc}"
      discard_last_experiment
    else
      echo ">>> Results: win_rate=${wr} avg_min_score=${ms}"

      # Compare with best
      improved=$(python3 -c "print('yes' if float('${wr}') > float('${best_win_rate}') else 'no')")
      if [[ "$improved" == "yes" ]]; then
        echo ">>> IMPROVED! Keeping. (${best_win_rate} -> ${wr})"
        best_win_rate="$wr"
        log_result "$commit" "$wr" "$ms" "keep" "$desc"
      else
        echo ">>> No improvement (${wr} <= ${best_win_rate}). Discarding."
        log_result "$commit" "$wr" "$ms" "discard" "$desc"
        discard_last_experiment
      fi
    fi
  else
    echo ">>> TIMEOUT or crash (exit code $?)"
    tail -20 "$LOG" 2>/dev/null || true
    log_result "$commit" "0.000000" "0.0" "crash" "timeout/crash: ${desc}"
    discard_last_experiment
  fi

  echo ""
  echo ">>> Results so far:"
  cat "$TSV"
  echo ""
done

echo "=== DONE: ${MAX_RUNS} experiments ==="
cat "$TSV"
