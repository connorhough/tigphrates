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
TOKENS_JSON="tokens.json"
TIMEOUT=720  # 12 min kill timeout
TOURNAMENT_EVERY=8  # refresh persistent Elo via tournament every N experiments

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

if [[ ! -f "$TOKENS_JSON" ]]; then
  echo '{"runs":[]}' > "$TOKENS_JSON"
fi

best_win_rate="0.000000"
best_pool_rate="0.000000"

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
  local pwr=$(grep "^vs_pool_win_rate:" "$LOG" 2>/dev/null | awk '{print $2}' || echo "")
  echo "${wr:-CRASH} ${ms:-0.0} ${pwr:-None}"
}

# --- Helper: log to TSV ---
log_result() {
  local commit="$1" wr="$2" ms="$3" status="$4" desc="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$commit" "$wr" "$ms" "$status" "$desc" >> "$TSV"
}

# --- Helper: generate move traces for a kept commit. ---
generate_traces() {
  local commit="$1"
  local out="traces/${commit}"
  if [[ -d "$out" ]] && compgen -G "${out}/game_*.jsonl" >/dev/null; then
    echo ">>> Traces already exist for ${commit}, skipping generation."
    return 0
  fi
  python3 python/play_traced.py \
    --out-dir "$out" \
    --games-vs-heuristic "${TRACED_VS_HEURISTIC:-3}" \
    --games-vs-champion "${TRACED_VS_CHAMPION:-2}" \
    --max-turns "${TRACED_MAX_TURNS:-1200}" \
    >> "$LOG.traces" 2>&1
  local rc=$?
  prune_traces
  return $rc
}

# --- Helper: ask the expert agent for a critique. ---
run_expert_agent() {
  local commit="$1"
  local traces_dir="traces/${commit}"
  local critique="${traces_dir}/critique.md"
  local shaping="models/shaping_config.json"
  local num_games
  num_games=$(ls "$traces_dir"/game_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  [[ "$num_games" -eq 0 ]] && return 1
  local prompt
  prompt=$(sed \
    -e "s|{{TRACES_DIR}}|$traces_dir|g" \
    -e "s|{{NUM_GAMES}}|$num_games|g" \
    -e "s|{{SHAPING_CONFIG_PATH}}|$shaping|g" \
    -e "s|{{CRITIQUE_PATH}}|$critique|g" \
    prompts/rl_expert.md)
  claude --print --output-format text \
    --model claude-sonnet-4-6 \
    --allowedTools Read,Glob,Grep \
    -p "$prompt" > "$critique" 2>> "$LOG.expert"
  [[ -s "$critique" ]]
}

# --- Helper: ask the DS agent to edit train.py. ---
run_ds_agent() {
  local commit="$1"
  local critique="traces/${commit}/critique.md"
  local shaping="models/shaping_config.json"
  [[ -s "$critique" ]] || return 1
  local prompt
  prompt=$(sed \
    -e "s|{{CRITIQUE_PATH}}|$critique|g" \
    -e "s|{{SHAPING_CONFIG_PATH}}|$shaping|g" \
    -e "s|{{RESULTS_TSV}}|$TSV|g" \
    prompts/rl_ds.md)
  claude --print --output-format text \
    --model claude-sonnet-4-6 \
    --allowedTools Edit,Read,Bash,Grep,Glob \
    -p "$prompt" >> "$LOG.ds" 2>&1
  # The DS agent commits its own edit. We just check that something
  # actually changed since the trainer's commit.
  ! git diff --quiet HEAD~1 HEAD -- python/train.py
}

# --- Helper: keep only the most recent N trace directories. ---
prune_traces() {
  local keep="${TRACE_RETENTION:-10}"
  if [[ ! -d traces ]]; then return 0; fi
  local dirs
  mapfile -t dirs < <(ls -1dt traces/*/ 2>/dev/null || true)
  local n="${#dirs[@]}"
  if (( n <= keep )); then return 0; fi
  local i
  for (( i=keep; i<n; i++ )); do
    rm -rf "${dirs[$i]}"
  done
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

  local claude_json
  claude_json=$(claude --print --output-format json --model claude-sonnet-4-6 --allowedTools Edit,Read,Bash,Grep,Glob -p "$prompt")

  # Log token usage to JSON
  local ts input_tokens output_tokens cache_read cache_creation cost_usd
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  input_tokens=$(echo "$claude_json" | python3 -c "import sys,json; u=json.load(sys.stdin)['usage']; print(u.get('input_tokens',0))")
  output_tokens=$(echo "$claude_json" | python3 -c "import sys,json; u=json.load(sys.stdin)['usage']; print(u.get('output_tokens',0))")
  cache_read=$(echo "$claude_json" | python3 -c "import sys,json; u=json.load(sys.stdin)['usage']; print(u.get('cache_read_input_tokens',0))")
  cache_creation=$(echo "$claude_json" | python3 -c "import sys,json; u=json.load(sys.stdin)['usage']; print(u.get('cache_creation_input_tokens',0))")
  cost_usd=$(echo "$claude_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_cost_usd',0))")

  python3 - <<PYEOF
import json, sys
with open("$TOKENS_JSON") as f:
    data = json.load(f)
data["runs"].append({
    "ts": "$ts",
    "run": $run,
    "branch": "$BRANCH",
    "input_tokens": $input_tokens,
    "output_tokens": $output_tokens,
    "cache_read_input_tokens": $cache_read,
    "cache_creation_input_tokens": $cache_creation,
    "cost_usd": $cost_usd,
})
# running totals
totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "cost_usd": 0.0}
for r in data["runs"]:
    for k in totals:
        totals[k] += r.get(k, 0)
data["totals"] = totals
with open("$TOKENS_JSON", "w") as f:
    json.dump(data, f, indent=2)
PYEOF

  echo ">>> Tokens: input=${input_tokens} output=${output_tokens} cache_read=${cache_read} cost=\$${cost_usd}"
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
  elif [[ -f ".autoresearch_ds_pending" ]]; then
    echo ">>> Using edit produced by DS agent in previous iteration."
    rm -f ".autoresearch_ds_pending"
  else
    echo ">>> Asking Claude for next experiment (fallback path)..."
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
    read -r wr ms pwr <<< "$(extract_results)"

    if [[ "$wr" == "CRASH" ]]; then
      echo ">>> CRASH (no results in log)"
      tail -20 "$LOG"
      log_result "$commit" "0.000000" "0.0" "crash" "crashed: ${desc}"
      discard_last_experiment
    else
      echo ">>> Results: win_rate=${wr} avg_min_score=${ms} vs_pool=${pwr}"

      # Improvement criterion: prefer vs-pool win rate when available; fall
      # back to vs-heuristic. Once a pool exists, league progress is the
      # signal that matters for self-play "better and better."
      if [[ "$pwr" != "None" && "$pwr" != "" ]]; then
        improved=$(python3 -c "print('yes' if float('${pwr}') > float('${best_pool_rate}') else 'no')")
        criterion="vs_pool"
        prev="$best_pool_rate"
        new="$pwr"
      else
        improved=$(python3 -c "print('yes' if float('${wr}') > float('${best_win_rate}') else 'no')")
        criterion="win_rate"
        prev="$best_win_rate"
        new="$wr"
      fi

      if [[ "$improved" == "yes" ]]; then
        echo ">>> IMPROVED on ${criterion} (${prev} -> ${new}). Keeping."
        best_win_rate="$wr"
        if [[ "$pwr" != "None" && "$pwr" != "" ]]; then
          best_pool_rate="$pwr"
        fi
        log_result "$commit" "$wr" "$ms" "keep" "$desc (vs_pool=${pwr})"
        if [[ -f "models/policy_final.pt" ]]; then
          cp "models/policy_final.pt" "models/policy_best.pt"
          echo ">>> Saved best model to models/policy_best.pt"
        fi
        # New: trace + expert + DS chain. Each step has a fallback so a
        # failure here does not wedge the loop.
        if generate_traces "$commit"; then
          if run_expert_agent "$commit" && run_ds_agent "$commit"; then
            echo ">>> Trace+expert+DS chain succeeded; next iteration will run the new edit."
            touch ".autoresearch_ds_pending"
            continue  # skip the legacy ask_claude_for_edit at top of next iter
          else
            echo ">>> Expert/DS agent failed; falling back to legacy direct-edit path."
          fi
        else
          echo ">>> Trace generation failed; falling back to legacy direct-edit path."
        fi
      else
        echo ">>> No improvement on ${criterion} (${new} <= ${prev}). Discarding."
        log_result "$commit" "$wr" "$ms" "discard" "$desc (vs_pool=${pwr})"
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

  # Periodic league refresh: a real round-robin keeps pool ratings honest
  # rather than relying on each new agent's per-opponent eval (which only
  # touches the opponent it played, leaving stale ratings on the rest).
  if (( run % TOURNAMENT_EVERY == 0 )) && [[ -d "models/pool" ]]; then
    echo ">>> Tournament refresh ($(ls models/pool/policy_*.pt 2>/dev/null | wc -l | tr -d ' ') members)..."
    python3 python/tournament.py --games-per-pair 2 --max-turns 1200 \
      >> "$LOG.tournament" 2>&1 || echo ">>> Tournament failed (see $LOG.tournament)"
    tail -20 "$LOG.tournament" 2>/dev/null || true
    echo ""
  fi
done

echo "=== DONE: ${MAX_RUNS} experiments ==="
cat "$TSV"
