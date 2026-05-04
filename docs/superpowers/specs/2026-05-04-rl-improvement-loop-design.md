# Agent-driven RL improvement loop

**Date:** 2026-05-04
**Status:** Design — not yet implemented
**Owner:** connor

## Problem

The existing autoresearch loop (`python/run_experiments.sh`) trains a model, scrapes win_rate from logs, and asks Claude to propose the next `train.py` edit. The signal Claude gets is a single scalar metric. It cannot see *how* the model is playing — only whether it wins more or less than before.

Result: the loop tweaks hyperparameters and reward coefficients somewhat blindly. There is no feedback path from observed in-game behavior ("the model never builds monuments"; "the model places leaders in spots that immediately get captured") back to the training signal.

## Goal

After a kept training run, generate rich per-move traces, have a T&E-expert Claude agent critique the games, then have a data-science Claude agent translate that critique into a concrete `train.py` edit. Loop.

## Non-goals

- Browser-based replay (Playwright). The headless runner is sufficient.
- Engineer/supervisor agent. Defer until the loop has been observed wandering.
- New visualization tooling. Move traces are JSONL; expert reads them as text.
- Replacing or refactoring the existing autoresearch infrastructure. Extend it.

## Architecture

Five components, four of them already exist or are minor extensions:

```
                ┌──────────────────────────────────────────────┐
                │ run_experiments.sh (orchestrator, extended)  │
                └───────────────────┬──────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐
   │  train.py        │  │  play_traced.py    │  │  expert agent    │
   │  (existing)      │  │  (new)             │  │  (Claude task)   │
   │  → checkpoint    │  │  → traces/*.jsonl  │  │  → critique.md   │
   └──────────────────┘  └────────────────────┘  └────────┬─────────┘
              │                     ▲                     │
              │                     │ uses                │
              │                     │                     ▼
              │                  bridge          ┌──────────────────┐
              │                                  │  DS agent        │
              │                                  │  (Claude task)   │
              │                                  │  → train.py edit │
              │                                  └──────────────────┘
              │                                           │
              └────────────── git commit ◄────────────────┘
```

### Component 1 — Trainer (existing)

`python/train.py`. No structural changes. Continues to write `models/policy_best.pt` on win_rate improvement and append a row to `results.tsv`.

The orchestrator's "kept" decision is unchanged from the existing `run_experiments.sh` flow.

### Component 2 — Trace generator (new)

**File:** `python/play_traced.py`

**Purpose:** play N headless games using the freshly trained checkpoint, dumping a rich trace per move.

**Reuse:** drives the existing TS bridge via the same RPC layer used by `evaluate.py` and `tigphrates_env.py`. Specifically, it calls `policy_server.py`-style inference but adds telemetry capture.

**Output format:** one JSONL file per game at `traces/<commit>/game_<i>.jsonl`. Each line is one move:

```json
{
  "turn": 12,
  "active_player": 0,
  "model_player": true,
  "phase": "action",
  "board": [[...], ...],          // 11x16 cell summary, JSON-friendly
  "hand": ["red","red","blue","green","black","black"],
  "scores": [3, 5, 2, 1],
  "leaders_placed": {"red": [3,4], "blue": null, ...},
  "legal_action_count": 47,
  "type_head_top5": [{"type":"placeLeader","prob":0.61}, ...],
  "param_head_top5_for_chosen_type": [{"action":"placeLeader red @(2,3)","prob":0.31}, ...],
  "value_estimate": 0.18,
  "chosen_action": {"type":"placeLeader","color":"red","row":2,"col":3,"argmax":true,"sampled":true},
  "immediate_reward": 0.05,
  "shaping_breakdown": {"leader_place":0.05,"kingdom_form":0.10,"score_delta":0.0}
}
```

The hierarchical action head (post-Phase 11.1) means we log top-K from `type_head` separately from `param_head`. Logging both `argmax` and the actual `sampled` action covers training/inference divergence.

**Match config:** by default plays:
- 3 games vs heuristic (the eval baseline)
- 2 games vs the prior champion in the pool (the model the new checkpoint is supposed to dethrone)

Configurable via env vars (`TRACED_VS_HEURISTIC=3`, `TRACED_VS_CHAMPION=2`).

### Component 3 — Expert reviewer agent (new)

**Invocation:** Claude Code subagent, dispatched stateless from `run_experiments.sh` (matching existing autoresearch pattern: `claude -p "<prompt>"` with file-based I/O). Use the `feature-dev:code-explorer` or `general-purpose` subagent type — to be decided in implementation; default to general-purpose for v1.

**Inputs (in the prompt):**
- `GAME_RULES.md` — full T&E rules
- `traces/<commit>/game_*.jsonl` — paths only; agent reads them via Read tool
- Current shaping config snapshot (parsed from `train.py` and env defaults): leader/kingdom/king/treasure/monument bonuses + decay schedule, score/margin/potential coefficients, BC_COEF
- Recent `results.tsv` tail (last ~10 runs) for context on what's been tried
- A short prompt prefix: "You are a Tigris & Euphrates expert. Critique the trained agent's play in the attached games. Focus on strategic blunders and recurring patterns. Reference specific turns. The model is **already** being rewarded for: <shaping config>. Do not suggest re-incentivizing things it's already incentivized to do — instead suggest what the rewards are missing or mis-weighted."

**Output:** prose markdown saved to `traces/<commit>/critique.md`. Free-form. Expected sections: recurring strategic patterns, individual blunders with turn references, hypotheses about why the model plays this way, what training signal might be missing.

**Cost envelope:** ~10K tokens input (5 games × ~80 moves × ~500 tokens compressed + rules + config), ~1K output. Roughly $0.10–0.20 per loop iteration at current Sonnet/Opus pricing.

### Component 4 — Data-science agent (new)

**Invocation:** Claude Code subagent, stateless. Same dispatch pattern as the existing single-Claude-edit-train.py call in the current `run_experiments.sh`.

**Inputs:**
- `traces/<commit>/critique.md`
- Current `train.py` (read by the agent itself)
- Recent `results.tsv` (so it can avoid re-trying recently failed ideas)
- Shaping config snapshot (same as expert)

**Task:** propose a single concrete edit to `python/train.py` that addresses the most pressing issue from the critique. Free-form translation — no constrained menu. The agent commits the edit with a descriptive message.

**Output:** `train.py` edit + git commit. (The orchestrator runs the next training pass against this commit; if win_rate drops, it gets reverted by the existing keep/discard logic.)

**Cost envelope:** ~5K input, ~1K output. Roughly $0.05–0.10 per iteration.

### Component 5 — Orchestrator (existing, extended)

**File:** `python/run_experiments.sh`

**Change:** after the existing "kept" branch (where it currently calls Claude to edit `train.py`), insert:

1. `python python/play_traced.py --commit "$COMMIT" --games-vs-heuristic 3 --games-vs-champion 2`
2. `claude -p "$(cat prompts/expert_prompt.md)" > traces/$COMMIT/critique.md` (stateless, with file paths in the prompt for the agent to Read)
3. `claude -p "$(cat prompts/ds_prompt.md)"` (stateless; agent edits and commits `train.py`)
4. Loop back to training.

If a step fails (timeout, crash, empty output), log and continue with a fallback: skip the agent step and use the existing direct-Claude-edit path.

## Data flow

```
1. train.py runs (5min budget) → models/policy_best.pt + results.tsv row
2. orchestrator checks win_rate delta
3. if kept:
     play_traced.py → traces/<commit>/game_*.jsonl
     expert agent reads traces + rules + shaping config → traces/<commit>/critique.md
     DS agent reads critique + train.py + results.tsv + shaping config → edits train.py, commits
   else:
     git reset --hard HEAD~1 (existing behavior)
4. loop
```

## Files to create / modify

**New:**
- `python/play_traced.py` — trace generator
- `prompts/expert_prompt.md` — expert agent system prompt template
- `prompts/ds_prompt.md` — DS agent system prompt template
- `traces/.gitignore` — ignore generated traces (`*` except `.gitignore`)

**Modify:**
- `python/run_experiments.sh` — insert trace + expert + DS calls after "kept" branch
- `python/policy_server.py` — expose a `/inference_with_telemetry` mode that returns top-K probs + value alongside the chosen action (or add this directly into `play_traced.py` if a separate code path is cleaner — to be decided in implementation)
- `python/train.py` — emit a small `shaping_config.json` alongside `policy_best.pt` so trace generator and agents can reference exact values used (vs reading env vars and defaults separately)

**No changes:** `evaluate.py`, `tournament.py`, `tigphrates_env.py`, `imitation_pretrain.py`, the TS bridge, the React UI.

## Error handling and failure modes

- **Trace generator crashes:** orchestrator logs and skips trace+critique steps; falls back to direct DS-agent edit on `results.tsv` only (graceful degradation).
- **Expert agent times out:** orchestrator logs and falls back to direct DS-agent edit (no critique).
- **DS agent produces a malformed `train.py`:** training run will crash → existing `run_experiments.sh` revert logic handles it.
- **DS agent produces no edit:** orchestrator skips the loop iteration.
- **Trace files grow large:** `traces/.gitignore` keeps them out of version control. Add a cleanup step that deletes traces older than the last 10 commits.
- **Hierarchical head mismatch:** if `train.py` reverts to a non-hierarchical head architecture, `play_traced.py` should detect and fall back to flat top-K logging.

## Testing

- **Unit:** `play_traced.py` against a known-good checkpoint, assert N games produced N JSONL files with valid schemas. Add to `python/tests/`.
- **Integration smoke:** run the full loop with `--dry-run` (existing `run_experiments.sh` flag) on a 30-second training budget. Verify traces are produced, expert agent returns non-empty critique, DS agent commits an edit.
- **No model-quality test for the agents.** Their value is measured by whether the loop's average win_rate improvement per iteration goes up. Track this in `results.tsv` and review after ~20 iterations.

## Risks

| Risk | Mitigation |
|---|---|
| Expert critiques are shallow ("the model should play better") | Few-shot exemplar critiques in the prompt template; emphasize specific turn references |
| DS-agent fixates on hyperparam tweaks ignoring critique | Prompt explicitly says "address the top issue from the critique"; track hyperparam-only iterations and warn if >50% over 10 runs |
| Trace gen + 2 agent calls add ~2 min per loop iteration | Acceptable — current loop is already ~5–8 min per iter dominated by training |
| Training/inference action divergence (sampled vs argmax) confuses expert | Log both; prompt the expert to focus on argmax for "what the model would do at evaluation time" |
| Cost grows with N games | Cap traces at 5 games per iteration; expert reads them sequentially or in summary |
| Cumulative trace storage | Cleanup script keeps last 10 commits' worth |

## Open questions

- Should the expert agent be allowed to ask follow-up questions (multi-turn)? **Decision: no, single-shot for v1.** Multi-turn risks context accumulation that hurts the autoresearch token discipline.
- Should the DS agent run sweeps or just propose single edits? **Decision: single edit per iteration.** Sweeps are already handled by `sweep.py`; this loop is for directed exploration.
- Should the loop checkpoint a "best critique" / "best DS edit" for retrospective analysis? **Defer.** Easy to add later; keep `traces/<commit>/` directories around as a poor man's audit log.

## Success criteria

After 20 iterations of the new loop running on a fresh branch:
- Average win_rate gain per iteration is higher than the existing scalar-only loop's recent history
- DS-agent edits demonstrably reference critique observations in their commit messages (qualitative check)
- No more than 20% of iterations fall into the fallback path (i.e., the agent stack is stable enough to be load-bearing)
