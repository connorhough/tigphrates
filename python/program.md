# autoresearch — Tigris & Euphrates RL

## Two modes of operation

### Mode 1: Script-driven (recommended — 80% fewer tokens)

`./python/run_experiments.sh` automates the loop. Claude is called ONLY to edit train.py.
Each Claude call is stateless (~2K tokens in, ~500 out). No context accumulation.

```bash
./python/run_experiments.sh --tag apr6 --max-runs 100
```

The script handles: run → grep results → log TSV → git keep/discard → call Claude for next edit → repeat.

### Mode 2: Fully autonomous (original autoresearch pattern)

For this mode, follow the instructions below. More flexible but ~10x more tokens.

---

## Context (read once at setup)

Game: Tigris & Euphrates. 2-4p board game on 11x16 grid. Imperfect info (hidden hands), stochastic (bag draws), min-score objective (score = min of 4 VP colors). ~1728 discrete actions with masking. Sparse rewards.

Files:
- `python/train.py` — ONLY file to edit. PPO training, architecture, hyperparams.
- `python/evaluate.py` — READ ONLY. Eval harness: 50 games vs heuristic AI.
- `python/tigphrates_env.py` — READ ONLY. Game env bridge.
- `GAME_RULES.md` — Full game rules.
- `src/ai/simpleAI.ts` — Heuristic opponent. Greedy, no catastrophes, simple conflicts.

Metric: `win_rate` (higher=better). Extract: `grep "^win_rate:" run.log`

Constraints: torch, numpy, gymnasium only. 5-min training budget. No other file edits.

## Experiment loop

```
LOOP FOREVER:
  1. Edit python/train.py (one idea)
  2. git commit
  3. python python/train.py > run.log 2>&1
  4. grep "^win_rate:\|^avg_min_score:" run.log
  5. Log to results.tsv (tab-sep: commit win_rate avg_min_score status description)
  6. win_rate up? keep. else git reset --hard HEAD~1
  NEVER STOP. NEVER ASK TO CONTINUE.
```

Crash? `tail -50 run.log`. Fix if trivial, skip if not.

## Ideas (priority order)

**Reward shaping** (biggest lever when win_rate=0):
- +bonus balanced scores (entropy of color scores)
- +bonus per VP gained, scaled by which color is lowest
- Δ(min_score) shaping already exists but coefficient too low (0.01) — try 0.1-0.5

**Hyperparams:**
- Rollout: try 256, 512 (game is long-horizon)
- LR: 1e-4 to 1e-3
- Entropy: 0.001 to 0.05
- Gamma: 0.995 or 0.999 (sparse rewards need high discount)
- PPO epochs: 4, 8, 16

**Architecture:**
- Deeper CNN (3-4 layers, 64ch)
- Residual connections
- Separate policy/value trunks

**Training:**
- LR schedule (warmup + cosine)
- Imitation pretrain from heuristic AI traces, then PPO finetune
- Population self-play
