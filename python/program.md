# autoresearch — Tigris & Euphrates RL

This is an experiment to have the LLM autonomously research and improve RL agents for Tigris & Euphrates.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files**: The relevant files are:
   - `python/program.md` — this file. Autonomous agent instructions.
   - `python/evaluate.py` — fixed evaluation harness, metrics, constants. **Do not modify.**
   - `python/tigphrates_env.py` — game environment bridge. **Do not modify.**
   - `python/train.py` — the file you modify. Policy network, PPO training, hyperparameters.
   - `GAME_RULES.md` — the complete rules of the game. Read this to understand what good play looks like.
   - `src/ai/simpleAI.ts` — the heuristic AI you're training against. Read to understand its strategy and weaknesses.
4. **Install dependencies**: `pip install -r python/requirements.txt` and ensure `torch` is installed.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The Game

Tigris & Euphrates is a 2-4 player strategic board game (Reiner Knizia, 1997). Key properties that affect RL training:

- **Imperfect information**: Players have hidden hands (6 tiles, drawn from bag)
- **Stochastic**: Random tile draws from bag
- **Min-score objective**: Final score = minimum across 4 VP colors. This means balanced scoring is critical.
- **Large action space**: ~1,728 discrete actions with masking
- **Sparse rewards**: Most reward comes at game end
- **Conflicts**: Internal (revolts) and external (wars) create sharp strategic moments

The heuristic AI (`simpleAI.ts`) is decent but predictable — it greedily places tiles to score, doesn't use catastrophes, and has simple conflict support logic.

## Experimentation

Each experiment trains on CPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding evaluation). You launch it as: `python python/train.py`.

**What you CAN do:**
- Modify `python/train.py` — this is the only file you edit. Everything is fair game: network architecture, optimizer, hyperparameters, reward shaping, rollout strategy, exploration techniques, training algorithm (you can replace PPO with anything).

**What you CANNOT do:**
- Modify `python/evaluate.py`. It is read-only. It contains the fixed evaluation function.
- Modify `python/tigphrates_env.py`. The environment interface is fixed.
- Modify game engine files (`src/`). The game rules are fixed.
- Add new Python dependencies beyond torch, numpy, gymnasium.

**The goal is simple: get the highest win_rate against the heuristic AI.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything in `train.py` is fair game: change the architecture, the algorithm, the hyperparameters, the reward shaping, the exploration strategy. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. A simplification that gets equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
win_rate:          0.120000
avg_min_score:     2.3
avg_margin:        -15.2
training_seconds:  300.1
total_seconds:     345.9
num_episodes:      412
num_steps:         52736
num_params:        234567
```

You can extract the key metric from the log file:

```
grep "^win_rate:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	win_rate	avg_min_score	status	description
```

1. git commit hash (short, 7 chars)
2. win_rate achieved (e.g. 0.120000) — use 0.000000 for crashes
3. avg_min_score (e.g. 2.3) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	win_rate	avg_min_score	status	description
a1b2c3d	0.120000	2.3	keep	baseline PPO
b2c3d4e	0.160000	3.1	keep	increase rollout to 256 steps
c3d4e5f	0.100000	1.8	discard	switch to A2C
d4e5f6g	0.000000	0.0	crash	triple model size (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr6`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `python/train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python python/train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^win_rate:\|^avg_min_score:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If win_rate improved (higher), you "advance" the branch, keeping the git commit
9. If win_rate is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~7 minutes total (5 min training + ~2 min eval). If a run exceeds 12 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo), fix it and re-run. If the idea itself is fundamentally broken, skip it.

## Ideas to explore

Here are research directions roughly ordered by expected impact:

### Quick wins (try first)
- Tune rollout length (64, 128, 256, 512)
- Tune learning rate (1e-4 to 1e-3)
- Tune entropy coefficient (0.001 to 0.05)
- Increase PPO epochs (4, 8, 16)
- Adjust discount factor for this sparse-reward game (0.995, 0.999)

### Reward shaping
- Bonus for balanced scores (entropy of score distribution)
- Bonus for placing leaders / joining kingdoms
- Penalty for losing conflicts
- Shaped reward: Δ(min score) per action instead of only terminal
- Curriculum: start with more shaping, anneal to sparse

### Architecture
- Deeper/wider CNN for board (3-4 layers, 64-128 channels)
- Residual connections in the trunk
- Separate value/policy trunks after shared encoder
- Attention over board positions
- Encode hand as bag-of-counts + normalize

### Training strategy
- Larger rollouts for this long-horizon game
- Learning rate schedule (warmup + cosine decay)
- Separate learning rates for encoder vs heads
- Multiple parallel environments (vectorized rollout collection)
- Prioritized experience: weight episodes by margin of defeat

### Advanced
- LSTM/GRU for temporal modeling across turns
- Population-based self-play (train against past checkpoints)
- Imitation learning: collect heuristic AI trajectories, pretrain policy, then finetune with PPO
- Monte Carlo Tree Search at inference time

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the game rules, re-read the heuristic AI code, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
