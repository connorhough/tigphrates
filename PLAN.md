# Tigphrates Improvement Plan

Two goals drive this plan:

1. **Competent AI opponent** — a human can play vs the AI and feel challenged.
2. **Self-improving trained agents** — RL pipeline can produce models that beat each other and the heuristic.

The plan is split into phases. Phase 1 (this session) targets goal 1 and unblocks goal 2. Subsequent phases are documented to preserve direction across sessions.

---

## Phase 1 — AI Competence Pack (this session)

Scope: heuristic AI bug fixes + leader rule gap + UI swap fix + policy injection point.

### 1.1 Fix AI conflict commit (attacker hand burn)

`src/ai/simpleAI.ts:54-59`. When attacker, AI commits all matching tiles even when defender has zero. Replace blind dump with bounded estimate of defender's max possible commitment, then commit the minimum to beat that ceiling.

Approach:
- Compute `defenderMaxPossible = defenderStrength + (defender hand worst-case for the conflict color)`.
- The attacker can't see the opponent hand, but can bound it: defender's conflict-color tiles ≤ `min(defenderHandSize, totalConflictColorInBag+hands)`. Use `defender.hand.length` as upper bound (cheap, board-state-derivable).
- Commit minimum to exceed `defenderMaxPossible` if achievable; else commit zero (don't waste).

### 1.2 Fix war order selection (use board, not hand)

`src/ai/simpleAI.ts:100-116`. Score each pending war color by `attackerStrength - defenderStrength + handCount(color)`. Pick the highest. If all negative, pick least-bad.

Implementation: simulate `setupWarConflict` for each pending color (or compute equivalent strengths in-place) to get base strengths, then add hand count.

Caveat: `setupWarConflict` mutates state. Cheaper approach: the strengths in `pendingConflict` are not yet populated for non-current war color. Compute strength from the unification-tile-position per color: count matching color tiles in each adjacent kingdom for attacker (player who placed the unifier) and defender side.

Simpler proxy that's still better than current: pick the color where `handCount > expectedDefenderCommit`. Use defender hand size as proxy.

### 1.3 Treasure-targeting heuristic

Add weight to AI placement decisions:
- Leader placement: prefer kingdoms with treasures (especially for green leader).
- Tile placement: prefer placing tiles that grow a kingdom containing the AI's green leader and treasures.

Implementation: in `tryPlaceLeader` and `tryPlaceScoringTile`, score each candidate placement with a small heuristic that counts adjacent-kingdom treasures and breaks ties on that.

### 1.4 Smarter tile placement (anti-corner)

`tryPlaceAnyTile` at `simpleAI.ts:277` picks the literally first valid cell. Replace with: prefer placements adjacent to existing tiles (extend kingdoms, build toward 2x2 monument squares). Avoid one-tile islands in remote corners.

Approach: score each placement by `# adjacent-tile-or-leader neighbors`, descending. Tiebreak by Manhattan distance to closest existing kingdom of any color.

### 1.5 Implement leader repositioning rule

GAME_RULES.md allows leader to be repositioned from on-board to a new square. Currently `placeLeader` (`actions.ts:319`) throws if leader is on board.

Approach:
- Update `handlePlaceLeader` to first remove the leader from its current cell if it has a position, then place at new position.
- Update `getValidLeaderPlacements` (validation.ts) — current logic already returns valid empty cells; should still work.
- Add tests: revolt triggered by repositioning into kingdom that has same-color leader; valid reposition without revolt.

### 1.6 UI partial swap

`App.tsx:95-102` and `HandPanel.tsx`. Replace "swap all hand" with selectable swap mode.

Approach:
- New state in `useGame` (or App): `swapMode: boolean`, `swapSelection: number[]` (hand indices).
- Swap button in `ActionBar` enters swap mode. While in swap mode, hand panel buttons toggle individual hand indices into selection. Swap button label changes to "Swap N" — pressing again commits selection (or pressing the all-tiles selection commits all).
- Hand panel currently groups by color count. For partial swap, present individual tile slots (not color counts). Easy: render `hand` array as one button per index when `swapMode` is true; otherwise show grouped color counts.

### 1.7 Policy injection point

`src/hooks/useGame.ts:64-75` hardcodes `getAIAction`. Add injection.

Approach:
- `useGame(opts: { getAction?: (state) => Promise<GameAction> | GameAction })`.
- Default to `getAIAction` when not provided.
- Caller (App.tsx) can later swap in a bridge call to a trained policy server.

This lands the hook even though no remote policy yet exists. Cost: ~10 lines.

---

## Phase 2 — RL Self-Play Foundation (this session — done)

Scope: turn the training pipeline into a real self-improving system.

### 2.1 Self-play opponent injection — DONE

`TigphratesEnv` gained `opponent_policy` parameter (`python/tigphrates_env.py`). When provided, replaces the heuristic AI for non-agent players. `_advance_ai_turns` calls the injected policy via the bridge; otherwise falls back to bridge `ai_action`. The standalone `TigphratesMultiAgentEnv` is left in place for future use; the pragmatic path was to extend the existing env.

### 2.2 Checkpoint pool (league self-play) — DONE

`models/pool/policy_*.pt` is populated:
- Initial random policy seeded as `policy_init.pt` at training start.
- Every 4 rollouts a snapshot is added.
- Final snapshot saved at end of run.
- Pool capped at `POOL_MAX_SIZE = 12`; oldest pruned.

`sample_opponent_policy(model)` mixes opponents per rollout:
- 25% heuristic (anchor)
- ~50% pool sample (recent past selves)
- ~25% current self (frozen snapshot)

### 2.3 Vs-pool win-rate eval — DONE

`evaluate.py:evaluate_vs_pool(policy_fn, opponent_loader, opponent_paths, games_per_opponent)` plays N games vs each pool member, returns aggregate `vs_pool_win_rate` plus per-opponent breakdown. Train.py runs this at end of training and emits `vs_pool_win_rate:` line.

`run_experiments.sh` improvement criterion: prefers `vs_pool_win_rate` when present, falls back to `win_rate` (vs heuristic) when pool is empty. Tracks `best_pool_rate` separately from `best_win_rate`.

Note: not full Elo (no rating math); win-rate vs pool is the gating metric. Real Elo can be Phase 3.

### 2.4 Strip reward double-counting — DONE

`src/bridge/server.ts:computeReward` now returns terminal ±1.0 only; intermediate steps return 0. Python-side shaping (`SCORE_DELTA_COEF`, `MARGIN_DELTA_COEF`) owns intermediate reward exclusively.

`tigphrates_env.py:step` no longer adds opponent-turn rewards to the agent's reward (the loop in `_advance_ai_turns` discards them). When the game ends during opponent turns, terminal reward is synthesized from final scores.

### 2.5 Observation channel fixes — PARTIAL

`encodeBoard` in `src/bridge/server.ts` rewritten: 15 channels (was 13).
- Ch 0-3: tile color (binary).
- Ch 4-7: leaders by COLOR (4 channels), value = `1 + ownerPlayerIndex`. Replaces dynasty-keyed encoding so the policy reads color AND owner directly.
- Ch 8: monument color1 index. Ch 9: monument color2 index. Ch 10: monument owner (best-effort from adjacent leader).
- Ch 11-14: catastrophes / treasures / river / flipped (was 9-12).

`BOARD_CHANNELS` in `tigphrates_env.py` updated to 15.

**Deferred (Phase 3):** ordered-hand encoding for swap action position-indexing. Hand is still encoded as color counts; swap actions are still effectively random over hand positions.

### 2.6 Vectorized envs — DEFERRED to Phase 3

Single env still. `NUM_ENVS=4` crash from prior experiments not investigated. Self-play league is in place; throughput is the next bottleneck.

---

## Phase 3 — Polish + Bridge Goals (this session)

### 3.1 Catastrophe in heuristic AI — DONE

`src/ai/simpleAI.ts:tryPlaceCatastrophe`. Triggers only when an opponent kingdom holds ≥3 face-up tiles of a color the AI has no leader for, and the candidate tile passes engine validity (no treasure, no adjacent leader). Skips cells adjacent to my own leaders. Slot ordering: priority between leader placement and swap fallback.

### 3.2 Hard AI mode in setup — DEFERRED

Requires either ONNX export + browser inference (small lift) or a local Python policy HTTP server (medium lift). The policy injection point from Phase 1.7 (`useGame({ getAIAction })`) is already the bridge — needs a remote-policy adapter on the browser side.

Recommended Phase 4 path:
- Add `python/policy_server.py` — minimal HTTP server, loads `models/policy_best.pt`, exposes `/action` taking serialized game state, returns action.
- Add a `RemotePolicyAdapter` in `src/ai/remotePolicy.ts` that POSTs game state and returns a `GameAction`.
- Setup screen toggle: `Heuristic` / `Trained (server)`.

### 3.3 Engine test coverage gaps — DONE

- King-as-wildcard scoring test was already present (`placeTile.test.ts:40`).
- Catastrophe stranding test was already present (`catastrophe.test.ts:96`).
- Added: `revolt.test.ts` "repositioning into kingdom with same-color enemy leader triggers revolt" — exercises the Phase 1.5 reposition path.

### 3.4 Surface leader repositioning in UI — DONE

`src/components/ActionBar.tsx`. On-board leaders now render as a `Move | Withdraw` button group. Tapping `Move` selects that leader as the placement target; existing board-click handler in `App.tsx` dispatches `placeLeader` to the engine which performs the lift+place internally.

Caveat: `getValidLeaderPlacements` does not simulate the lift before computing valid destinations, so the UI may hide a small subset of legal reposition targets near the lifted leader's source kingdom. The engine is permissive — invalid clicks throw and `dispatch` swallows them. Acceptable for now; Phase 4 can add a reposition-aware validator.

### 3.5 Ordered-hand observation — DONE

`src/bridge/server.ts:encodeObservation` now emits `handSeq`: a 6-slot int array, value = colorIndex per hand position (0-3), -1 for empty slots. `python/tigphrates_env.py` exposes `obs["hand_seq"]` (shape `(6,)`, dtype int32). `train.py:PolicyValueNetwork` consumes it via 6×5 one-hot (5 = 4 colors + empty marker) → flatten to 30 features → concat into trunk input. `extra_dim` is now 73 (was 43).

The policy can now reason about which specific tile sits at each hand position, so position-indexed swap actions and commit-support actions carry meaningful signal.

### 3.6 Real Elo math — DONE

`evaluate.py:_compute_elo` iterates Elo updates over the per-opponent win rates (K=32, baseline=1500). Pool members are treated as fixed at 1500. Result is a single scalar that moves intuitively with overall win rate. Reported in training summary as `elo: <float>`. Aggregate `vs_pool_win_rate` is still the gating metric for `run_experiments.sh`; Elo is informational.

### 3.7 Vectorized envs — DEFERRED to Phase 4

Single env still. Throughput bottleneck on 1728-action sparse-reward game. Investigate why `NUM_ENVS=4` crashed in earlier experiments (likely subprocess stdout interleaving — each env should own its own bridge process).

---

## Phase 4 — Hard AI + Vectorization (this session — done)

### 4.1 Hard AI policy server — DONE

`python/policy_server.py` — minimal HTTP server, loads a saved `PolicyValueNetwork` checkpoint, owns a Node bridge subprocess. POST `/action` with `{ state, playerIndex }` returns `{ action, label, actionIndex }`.

`src/bridge/server.ts` gained two new RPC methods:
- `load_state(state)` → registers a serialized GameState as a new game; returns gameId.
- `decode_action(gameId, actionIndex)` → maps a flat action index to a concrete `GameAction` for dispatch.

`src/ai/remotePolicy.ts:getRemoteAIAction(state, url?)` — async client that POSTs the game state to the server. Throws on failure; caller falls back to heuristic.

`src/components/SetupScreen.tsx` — adds an `AI Difficulty` toggle (`Heuristic` / `Trained (server)`).

`src/App.tsx` — wires `aiKind` from setup screen into `useGame`'s policy. When `aiKind === 'hard'`, the `policy` callback calls `getRemoteAIAction`; on any error it logs a warning and falls back to the heuristic so the game never freezes.

End-to-end smoke: `curl POST /action` with a fresh 2p state returned `placeCatastrophe@3,12` (a valid action) at HTTP 200.

**Note:** The trained checkpoint must match the current observation schema. After Phase 3.5 (15-channel board + ordered hand), pre-existing `models/policy_best.pt` would not load. Re-train with `python python/train.py` to refresh.

### 4.2 Vectorized envs — DONE

`python/train.py:VecTigphratesEnv` — holds N independent `TigphratesEnv` instances (one Node bridge subprocess each). `collect_rollout_vec` does a single batched forward pass over all N envs each outer step, then steps each env sequentially with its sampled action. GAE is computed per env then concatenated.

Default `NUM_ENVS = 4`. Earlier `NUM_ENVS=4` crash from `results.tsv` row `c9e00ea` was from sharing a single subprocess across envs; the new structure spawns one bridge per env so that crash mode is gone.

Real wall-clock parallelism is currently bounded by Python sequential bridge calls (no threading yet). Future improvement: thread or asyncio the bridge calls so subprocess wait time overlaps. Even sequential, the batched forward pass is the main RL win.

### 4.3 Reposition-aware leader validation — DONE

`src/engine/validation.ts:getValidLeaderPlacements` now accepts optional `repositioningFrom: Position`. When set:
- The leader's source cell is virtually cleared via `boardWithoutLeaderAt` before computing kingdoms.
- The source cell itself is excluded from valid destinations.
- Temple-adjacency and unite-kingdom checks use the lifted board view.

`src/App.tsx:highlights` passes the leader's current position when the selected leader is on-board, so the UI shows accurate reposition destinations.

### 4.4 Persistent Elo per pool member — DONE

`python/evaluate.py`:
- `_load_elo_table` / `_save_elo_table` — JSON sidecar at `models/pool/elo.json`.
- `update_persistent_elo(pool_dir, per_opponent, games_per_opponent)` — applies Elo updates per game (not per opponent) to avoid one-sweep rating swings; mutates both agent and opponent ratings; persists.
- `ELO_AGENT_KEY = "_agent"` is the agent's running rating.

`python/train.py` calls `update_persistent_elo` after `evaluate_vs_pool` and emits `persistent_elo:` line in the summary.

The league now has memory across runs: future `run_experiments.sh` cycles can compare current rating vs prior best.

### 4.5 Imitation pretrain — DEFERRED to Phase 5

Documented for later. Bigger surgery, requires a trace-collection pipeline.

---

## Phase 5 — Imitation, Tournament, Threading (this session — done)

### 5.1 Threaded bridge calls — DONE

`python/train.py:VecTigphratesEnv` now owns a `ThreadPoolExecutor(max_workers=n_envs)`. `reset_all`, `action_masks`, and `step_all` map across the pool so each env's bridge subprocess `readline` happens concurrently — Python releases the GIL during the subprocess wait.

`collect_rollout_vec` was restructured to:
1. Snapshot pre-step shaping inputs.
2. Append per-env transition data.
3. Issue `step_all(actions)` — N envs step in parallel.
4. Compute shaped rewards from each env's new obs.

Smoke (`TIME_BUDGET=20`, `NUM_ENVS=4`): 9216 steps / 21s ≈ 439 steps/sec, vs ~264 steps/sec single-threaded. ~1.6× speedup at N=4 — sub-linear because each env still serializes its own opponent-policy chain internally.

### 5.2 Imitation pretrain from heuristic — DONE

`python/imitation_pretrain.py`. Plays `--games` self-play games where every player is `simpleAI`, records `(obs, action_index, mask)` per heuristic decision via the bridge, then trains a `PolicyValueNetwork` with masked cross-entropy loss against the heuristic's action distribution.

Smoke (`--games 8 --epochs 2`): 1551 transitions in 1.4s, BC training in 0.9s, accuracy 0.22 → 0.26 over two epochs. Saves `models/policy_bc.pt`.

`python/train.py` checks for `models/policy_bc.pt` at startup and loads it as warm start when present. When loaded, the message `Loaded BC warm start from models/policy_bc.pt` is printed before training begins.

### 5.3 Bridge `delete_game` — DONE

`src/bridge/server.ts:handleDeleteGame` discards a game from the in-memory `games` Map. `python/policy_server.py` calls it after each `/action` request so the server doesn't leak memory under sustained load.

### 5.4 ONNX export — DEFERRED to Phase 6

Documented for later. Bigger lift: needs `torch.onnx.export` on `PolicyValueNetwork` with the dict observation handled (`onnxruntime-web` doesn't accept dict inputs natively — flatten first), plus a browser inference adapter to replace `getRemoteAIAction`.

### 5.5 Tournament harness — DONE

`python/tournament.py`. Loads every checkpoint in `models/pool/`, plays round-robin with `--games-per-pair` games in each seat order. Prints leaderboard sorted by overall win rate, then updates `models/pool/elo.json` with symmetric pairwise Elo (`update_elo_pair` in `evaluate.py`). Final `_agent` rating is set to the top-of-pool rating so future runs of `train.py` start from a meaningful league position.

Smoke (4 contestants × 2 games × 2 seats = 24 games):

```
=== Tournament Leaderboard ===
  policy_final_r0009                0.750  (9/12)
  policy_init                       0.500  (6/12)
  policy_r0004                      0.500  (6/12)
  policy_r0008                      0.250  (3/12)
```

Optional `--include-heuristic` adds the built-in `simpleAI` as a contestant (with seat-symmetry inversion since the env doesn't natively support heuristic-as-protagonist). Diagnoses collapse / forgetting that agent-vs-pool eval can't see.

---

## Phase 6 — Mask audit, BC+PPO, multi-player, ONNX (this session — done)

### 6.1 ONNX export — DONE (browser inference deferred to Phase 7)

`python/export_onnx.py`. Wraps `PolicyValueNetwork` in `FlatPolicy` with a positional-tensor forward, exports to ONNX (opset 17). Mask is applied in-graph (`-1e9` penalty, not `-inf`, for portability). All inputs have a dynamic batch axis so callers can run batched inference.

Smoke: produced `models/policy.onnx` at 7.6 MB from a fresh `policy_best.pt`. Runtime requirement: `pip install onnx` (now installed in `python/venv/`); `onnxscript` not required because the legacy TorchScript exporter is used.

Browser inference (extracting encoder out of `src/bridge/server.ts`, adding `onnxruntime-web` to npm deps, wiring `src/ai/onnxPolicy.ts`) deferred to Phase 7.

### 6.2 Multi-player tournaments — DONE

`python/tournament.py` gained `--player-count` (2-4). With N>2, each match is "agent (seat 0) vs N-1 copies of opponent (seats 1..N-1)" — same opponent_policy controls all non-agent seats. Round-robin pair structure unchanged.

`play_match` final-score check now compares agent vs `max` over all opponents (was strict 2-player).

Smoke (`--player-count 3`, 4 contestants, 2 games per pair × 2 seats): 24 games in 10s, leaderboard sensible (ratings clustered 1469-1514 around baseline 1500).

### 6.3 BC + PPO joint loss — DONE

`python/tigphrates_env.py` gained `expert_action_index() -> int` — queries the bridge's `ai_action` for the current active player and returns a flat action index.

`collect_rollout_vec` records the heuristic's chosen action with probability `BC_QUERY_PROB = 0.25` per agent step, stored in `per_env_expert[i]` (-1 = not queried).

`ppo_update` adds `BC_COEF * F.cross_entropy(logits[bc_mask], expert_targets)` to the total loss for rows where the expert was recorded. Reported as `bc_loss` in the rollout summary.

Smoke: BC+PPO trained without crashing, 8192 steps in 22s with throughput ~372 steps/sec (was 439 without BC queries — expected overhead from 25% extra bridge calls).

### 6.4 Action-mask correctness audit — DONE

`python/train.py:ppo_update` now asserts (under `__debug__`) that every taken action in a minibatch is within its own mask. Catches off-by-one bugs in mask construction or stale masks after a phase-changing engine event. Cheap — one indexed lookup per row.

Smoke: full training run completed with no assertion failures, confirming the existing rollout collection produces consistent (action, mask) pairs.

### 6.5 Engine perf for tournament throughput — DEFERRED

Not needed at current pool sizes (≤12). Revisit when leaderboard regularly involves 30+ contestants.

---

## Phase 7 — Encoder, ONNX, scheduler, diversity (this session — done)

### 7.1a Bridge encoder extraction — DONE

`src/bridge/encoder.ts` holds the browser-safe encoding logic: `COLOR_INDEX`, `BOARD_CHANNELS`, `HAND_MAX`, `ACTION_SPACE_SIZE`, `EncodedAction`, `encodeBoard`, `encodeObservation`, `activePlayerIndex`, `enumerateValidActions`, `createActionMask`. No Node-only imports — pure functions of `GameState`.

`src/bridge/server.ts` shrunk from ~530 lines to ~250: now imports the encoder, keeps only Node-bound concerns (readline RPC loop, in-memory game map, game lifecycle handlers).

### 7.1b Browser ONNX policy — DONE

`onnxruntime-web` added to npm dependencies. `models/policy.onnx` copied to `public/policy.onnx` so Vite serves it as a static asset.

`src/ai/onnxPolicy.ts:getOnnxAIAction(state)`:
- Lazy-loads an `ort.InferenceSession` from `/policy.onnx` (cached in module scope).
- Encodes obs via the shared `encoder.ts` and flattens to typed-array tensors matching `python/export_onnx.py`'s input names.
- Runs the session, argmaxes the (already-masked) logits.
- Decodes the flat index back to a `GameAction` via the same `enumerateValidActions` used by the bridge.

`src/components/SetupScreen.tsx` gained a third `AIKind` value (`onnx`) with a "Trained (browser)" button. `src/App.tsx` routes `aiKind === 'onnx'` through `getOnnxAIAction`, falling back to the heuristic on any error so the game never freezes if ONNX fails to load.

Browser bundle impact: ~26 MB ONNX runtime WASM (gzip 6 MB), ~660 KB main JS (gzip 184 KB). Vite build succeeds.

### 7.2 Configurable player count in training — DONE

`evaluate.py:PLAYER_COUNT` now reads from `os.environ.get("PLAYER_COUNT", 2)`. `train.py` reuses it everywhere via the existing import. Smoke: `PLAYER_COUNT=3 python python/train.py` runs cleanly.

### 7.3 Pool diversity metric — DONE

`python/pool_diversity.py`:
- Plays heuristic self-play games until N obs samples are captured.
- Runs every pool member's forward pass over the sample.
- Computes pairwise mean **symmetric** KL between action distributions.
- Prints a similarity matrix and a redundancy ranking (lowest min KL = closest duplicate of another member).
- Optional `--prune-threshold` flag deletes near-duplicates (keeps the higher-Elo of each pair).

Smoke: confirmed identical-by-construction snapshots show KL=0.000; structurally different ones show KL ≈ 0.4-1.0.

### 7.4 Per-episode league scheduler — DONE

`python/train.py:LEAGUE_SCHEDULER = "per_episode"`. `collect_rollout_vec` accepts `on_reset(env_idx)` callback invoked after each env reset (initial reset and after `done`). Main loop binds it to `sample_opponent_policy(model)` so each env starts with its own opponent and resamples on episode end.

Per-rollout mode (`LEAGUE_SCHEDULER = "per_rollout"`) preserved as fallback. Logging shows comma-separated unique opponent labels per rollout.

### 7.5 Hyperparameter sweep — DEFERRED to Phase 8

Documented for later. Needs parallel run infra (CI matrix or local job queue) to be useful — sequential sweep at 5 min/run with 5+ knobs is too slow for the autoresearch cadence.

---

## Phase 8 — Mac mini local infra (this session — done)

Constraint: everything must run locally on a Mac mini (M-series Apple Silicon, 8-16 GB RAM, ~10 cores). No cloud, no multi-machine assumptions.

### 8.1 Hyperparameter sweep — DONE

`python/sweep.py`. Defines an editable `DEFAULT_GRID` of env-var overrides; spawns `train.py` once per cell. `--concurrency N` controls parallelism via `ProcessPoolExecutor` (Mac mini default = 1; recommended ceiling N ≤ cpu_count // 6 because each train.py spawns NUM_ENVS Node bridges).

Each run gets its own `RUN_DIR` so per-run pools never overlap. After all runs complete, prints a leaderboard sorted by `vs_pool_win_rate` and persists `sweep_results.tsv`.

Smoke (4 cells × 15s + eval = ~85s total): ranking emerged correctly; best cell `BC_COEF=0.0, POOL_OPPONENT_PROB=0.5` got vs_pool_win_rate 0.667.

### 8.2 Tournament-driven Elo refresh — DONE

`python/run_experiments.sh` now runs `python/tournament.py` every `TOURNAMENT_EVERY=8` experiments. Output goes to `run.log.tournament`. Keeps pool ratings honest as new members join — without the periodic refresh, only the agent vs latest-pool-member match touches Elo.

### 8.3 Apple Silicon MPS — wired but **disabled by default**

`_pick_device()` in `train.py` reads `TORCH_DEVICE` env override; defaults to CPU. Reason: MPS adds ~5-10s warmup + ~5ms per-call transfer overhead that *hurts* throughput at this batch size (NUM_ENVS=4 → B=4 forward passes). Smoke comparison at TIME_BUDGET=60s:

| Device | Steps    | Steps/sec |
|--------|----------|-----------|
| CPU    | ~7K-9K   | 327-372   |
| MPS    | ~7K      | 117       |

Bottleneck is bridge subprocess RPC, not the forward pass. MPS only wins once batch sizes grow (e.g., bigger model in 8.6) or rollout volume increases. Opt in with `TORCH_DEVICE=mps`.

All tensor-creation sites updated to honor `DEVICE`; checkpoint save/load uses `map_location="cpu"` so pool files stay portable.

### 8.6 Env-var model size — DONE (folded into 8.3)

`BOARD_CONV_CHANNELS`, `HIDDEN_DIM`, `NUM_HIDDEN_LAYERS`, plus all PPO/reward/league knobs (`LEARNING_RATE`, `BC_COEF`, `POOL_OPPONENT_PROB`, etc.) now read from environment with the existing values as defaults. Lets `sweep.py` traverse any combination without touching source.

`POOL_DIR` and `RUN_DIR` are also env-overridable, so concurrent sweeps don't stomp each other's pools / final models.

### 8.5 ONNX lazy load — DONE

`src/App.tsx` switched to `await import('./ai/onnxPolicy')` inside the policy callback so the 26 MB onnxruntime-web WASM only downloads when the user actually picks "Trained (browser)".

Vite build output before / after:
- Main bundle: 657 KB → **256 KB** (gzip 184 KB → **76 KB**).
- ONNX chunk: emitted as separate `onnxPolicy-*.js` (401 KB / gzip 109 KB), pulled lazily.
- WASM: still 26 MB but only fetched on first ONNX use.

The heuristic-only path is now fast on first load, matching the constraint that this all runs local on the Mac mini (smaller initial download → faster cold-start over local network for any device viewing the dev server).

### 8.4 Imitation pretrain at higher quality — DEFERRED to Phase 9

Documented for later. Needs an external demonstrator stronger than `simpleAI` to actually be an improvement. The natural source is the trained policy itself once it surpasses the heuristic.

---

## Phase 9 — Resilience + parallelism (this session — done)

### 9.1 Self-imitation pretrain — DONE

`python/imitation_pretrain.py` gained `--expert <path>` flag. When set, all per-step decisions come from the loaded `PolicyValueNetwork` (argmax over masked logits) instead of `bridge.ai_action`. Lets the next training run BC-pretrain from a strong trained demonstrator once the policy beats `simpleAI`.

Smoke (`--games 4 --epochs 1 --expert models/policy_final.pt`): produced 20K transitions in 30s, BC trained with 0.98 accuracy (expected — model imitates its own argmax).

### 9.2 Resumable training — DONE

`python/train.py` writes `<RUN_DIR>/checkpoint_resumable.pt` at the end of each run with `{model_state, optimizer_state, torch_rng, numpy_rng}`. Set `RESUME_FROM=<path>` to pick up where you left off — model + optimizer + RNG all restored. Bare `state_dict` files (just weights) also load with a fresh optimizer.

Use case: Mac mini sleep / restart in the middle of a long training session no longer wipes progress.

Smoke: ran 12s training → wrote checkpoint → resumed with `RESUME_FROM=...` → "Resumed from checkpoint_resumable.pt" message confirmed; second run continued from the saved state.

### 9.3 Async / pipelined bridge calls — DEFERRED to Phase 10

The naive Phase 5.1 ThreadPoolExecutor already overlaps per-env RPCs across N envs. Real pipelining (sending multiple requests then matching responses by id, or a `agent_step_heuristic` RPC that runs the opponent loop server-side) only helps in narrow cases (heuristic opponent only) and requires careful Python/Node semantic matching. Skipped for this session.

### 9.4 Replay buffer — DEFERRED to Phase 10

PPO is on-policy — adding a replay buffer is a substantial algorithmic shift (importance-sampled corrections, separate update phases). Documented for later.

### 9.5 Tournament concurrency — DONE

`python/tournament.py` gained `--concurrency N`. Match jobs are submitted to a `ThreadPoolExecutor`; each runs in its own thread with its own bridge subprocess. Recommended ceiling: `N <= cpu_count // 4` on a Mac mini (each match opens one bridge subprocess + the agent's internal model inference).

Smoke (2 contestants, concurrency=2): 2 matches finished in 1s. Pattern matches `VecTigphratesEnv` from Phase 5.1.

---

## Phase 10 — Throughput + curriculum + observability (this session — done)

### 10.1 Batched bridge `agent_step` — DONE

`src/bridge/server.ts:handleAgentStep`. New RPC that:
1. Applies the agent's action via `gameReducer`.
2. Runs `simpleAI` for every non-agent seat until control returns to the agent or the game ends.
3. Returns `{reward, done, obs, mask, info, activePlayer, turnPhase}` — everything Python needs for the next iteration in one round-trip.

`python/tigphrates_env.py:step` takes this fast path when `opponent_policy is None` (heuristic opponent). For trained-policy opponents it still falls back to the original step + `_advance_ai_turns` chain since Python must run model inference between bridge calls.

`action_mask()` reuses the mask the fast path returned, eliminating a second RPC per outer step. Mask cache invalidated on `reset()` and at end of slow-path step.

**Throughput** (TIME_BUDGET=20s):
| Mode | Steps | Steps/sec |
|------|-------|-----------|
| Mixed opponent (~25% heuristic) | 7168 | 341 |
| Heuristic-locked (curriculum 1.0) | **14336** | **717** |

Heuristic-locked training is **2.2× faster** — the autoresearch loop's baseline runs are exactly this case.

### 10.3 Curriculum schedule — DONE

`python/train.py:heuristic_prob_now(elapsed, total)`. Linear decay from `CURRICULUM_HEURISTIC_START` (default 1.0) to `CURRICULUM_HEURISTIC_END` (default 0.1) over the training budget. Disabled by default (`CURRICULUM_ENABLED=0`); set `CURRICULUM_ENABLED=1` to opt in.

`sample_opponent_policy(model, heuristic_prob=...)` now accepts a per-call override; main loop and the per-episode `on_reset` callback both pass the curriculum value, so the rollout opponent mix shifts over training time.

Combined with 10.1, early training runs at 717 steps/sec on heuristic-only, then transitions into the league as the agent stabilizes.

### 10.4 Plot generator — DONE

`python/plot_results.py`. Reads `results.tsv` from the autoresearch loop, renders either:
- A two-axis matplotlib PNG (`win_rate` blue, `avg_min_score` orange) — falls back to ASCII if matplotlib isn't installed; or
- Block-character sparklines (`--ascii`) for terminal-only viewing.

Smoke: ASCII spark on the existing 50-row `results.tsv` rendered the win_rate trend (peak 0.28 at experiment 14) and the avg_min_score progression. PNG version produces 80 KB image at 120 dpi.

### 10.5 Hierarchical action head — DEFERRED to Phase 11

The 1728-action space is mostly invalid at any state (mask sum ~30-100). A factored head (action-type → parameters) is the next architectural lever but requires substantial network and obs/mask rework. Documented for later.

### 10.2 V-trace replay — DEFERRED to Phase 11

Off-policy refactor too large for this session.

---

## Phase 11 — Hierarchical head + future work

### 11.1 Hierarchical action head — DONE

Two-stage policy: type head picks one of 10 action types; param head picks
within the chosen type's slot range. Per-state mask sums collapse from up
to 1728 down to ≤10 types and typically <50 valid params per type, so
entropy regularization no longer pushes mass into invalid regions.

Touch points:
- `src/bridge/encoder.ts` — added `ACTION_TYPES`, `NUM_ACTION_TYPES`,
  `TYPE_PARAM_SIZES`, `TYPE_BASES`, `decodeFlatAction()`. Each
  `EncodedAction` now carries `typeIdx` + `paramIdx`. Total flat layout
  unchanged (1728 slots; partition contiguous).
- `python/train.py` — `PolicyValueNetwork` split: `policy_head` removed,
  replaced by `type_head` (HIDDEN_DIM → 10) + `param_head` (HIDDEN_DIM →
  1728). `forward()` returns `(type_logits, param_logits, value)`.
  `hierarchical_dists()` builds the (B, NT, MP) padded view, masking with
  per-action mask + pad mask. `get_action_and_value` does hierarchical
  sample/log-prob/entropy. PPO `ppo_update` rewires log-prob via chain
  rule; entropy = H(type) + H(param | sampled-type) (MC-estimate).
- `_adapt_state_dict()` translates pre-11.1 `policy_head.{weight,bias}` →
  `param_head.{weight,bias}` so old pool/BC checkpoints still load
  (type_head starts fresh). `strict=False` everywhere lets the type_head
  init from random.
- `python/imitation_pretrain.py` — BC loss split into type CE + chosen-
  type param CE. Hierarchical argmax for self-imitation demonstrator path.
- `python/export_onnx.py` — `FlatPolicy` returns `(type_logits,
  param_logits, value)`. Per-action mask still applied to param_logits
  in-graph; type masking left to the JS caller.
- `src/ai/onnxPolicy.ts` — hierarchical argmax: build valid-type set from
  enumerated `EncodedAction` list, argmax masked `type_logits` over valid
  types, then argmax `param_logits` inside the chosen type's slot range.
- `python/policy_server.py`, `python/tournament.py`,
  `python/pool_diversity.py` — updated to load checkpoints via
  `_adapt_state_dict(...)` + `strict=False`. `pool_diversity` reconstructs
  the joint flat distribution for KL comparability across versions.

Verify (TIME_BUDGET=20s, NUM_ENVS=4, default 2-player):
| Metric              | Pre-11.1 baseline | Post-11.1 |
|---------------------|-------------------|-----------|
| Steps/sec           | 327-372           | ~292      |
| Mask audit          | passes            | passes    |
| vs_pool_win_rate    | 0.55-0.67         | 0.583     |
| persistent_elo      | ~1530             | 1533      |
| BC accuracy (4g/2e) | 0.22 → 0.26       | 0.21 → 0.27 |

Throughput regresses ~12% — the per-step gather + extra Categorical
construction adds overhead at this batch size (B=4). Real win is the
masked-down distributions; expect it to compound over longer runs as the
policy doesn't waste exploration on invalid types. Re-benchmark over a
60-90s budget once the new heads have warmed up.

`tests`: 177 engine tests still pass; `npx vite build` succeeds with the
ONNX chunk still split; `npm run lint` baseline 11 errors unchanged.

### 11.2 V-trace replay buffer — DONE

Off-policy sample reuse via a length-K deque of recent rollouts. Each PPO
update mixes the current rollout with every rollout in the buffer; replay
rows get a truncated importance weight `min(c_bar, pi_new(a|s)/pi_old(a|s))`
*before* PPO clipping, so stale data can't up-weight the gradient past the
trust region.

Touch points:
- `python/train.py` — `REPLAY_K = int(os.environ.get("REPLAY_K", 2))`,
  `REPLAY_RHO_BAR = 1.0`. `replay_buffer = deque(maxlen=REPLAY_K)` in the
  training loop.
- `combine_with_replay(current, replay_buf)` — concatenates obs/actions/
  log_probs/advantages/returns/masks/values/expert_actions across rollouts
  and tags each row with `is_replay`.
- `ppo_update` — accepts the `is_replay` tensor; for replay rows
  `effective_ratio = clamp(ratio, max=REPLAY_RHO_BAR)`, current rows keep
  the standard ratio. PPO clip [1-eps, 1+eps] then runs over the capped
  ratio for both populations. Value targets stay stale for replay rows
  (we don't recompute GAE under the current value head); at K=2 the bias
  is small and PPO clipping bounds policy drift.

Verify (TIME_BUDGET=20s, NUM_ENVS=4, REPLAY_K=2):
| Metric              | REPLAY_K=0 (11.1) | REPLAY_K=2 |
|---------------------|-------------------|------------|
| Steps/sec           | ~292              | ~228       |
| pg_loss             | -0.037 → -0.022   | -0.046 → -0.002 |
| vf_loss             | 21 → 40           | 57 → 4     |
| entropy H           | 4.36 → 3.84       | 4.31 → 3.64 |
| vs_pool_win_rate    | 0.583             | 0.500      |
| persistent_elo      | 1533              | 1523       |
| NaN / mask-audit    | none              | none       |

Throughput drops ~22% — each PPO update covers (K+1)× rows so it pays K+1
forward/backward passes per rollout. Each transition now contributes to up
to (K+1) updates, so sample efficiency is the upside; throughput is the
cost. Net win is empirical and depends on training budget; needs a longer
(60-90s+) head-to-head against REPLAY_K=0 to characterize.

Knobs to sweep next: `REPLAY_K ∈ {0, 2, 4}` × `REPLAY_RHO_BAR ∈ {0.5, 1.0,
2.0}`. 11.3 reward sparsity ablation should also probe these.

### 11.3 Reward sparsity ablation — DONE

`python/sweep.py:DEFAULT_GRID` set to `SCORE_DELTA_COEF ∈ {0.0, 0.5, 1.5}` ×
`MARGIN_DELTA_COEF ∈ {0.0, 1.0}`, 6 cells × 90s × eval_games=4. Sweep
results in `sweep_results_11_3.tsv`.

Leaderboard (vs_pool_win_rate desc, single-seed):

| SCORE_DELTA | MARGIN_DELTA | vs_pool | persistent_elo |
|-------------|--------------|---------|----------------|
| **0.0**     | **1.0**      | 0.857   | 1569.7         |
| **0.5**     | **0.0**      | 0.857   | 1566.0         |
| 0.0         | 0.0          | 0.714   | 1541.7         |
| 1.5         | 1.0 (current default) | 0.714 | 1540.4 |
| 1.5         | 0.0          | 0.667   | 1528.4         |
| 0.5         | 1.0          | 0.583   | 1512.0         |

Findings:
- Current defaults (1.5, 1.0) place 4th. The dense `SCORE_DELTA_COEF=1.5`
  shaping hurts more than it helps under hierarchical heads + replay.
- Two cells tied at the top: pure margin-shaping (`SCORE=0.0, MARGIN=1.0`)
  and modest score-shaping with no margin term (`SCORE=0.5, MARGIN=0.0`).
  Both ~30 elo above the 1.5/1.0 default.
- Pure terminal reward (`0.0, 0.0`) ties for 3rd at 0.714 — surprisingly
  competitive at this budget. Dense shaping is not free.
- Single-seed noise is real (~0.14 spread between cells) — the 0.857-tied
  pair is roughly tied within noise; treat both as reasonable defaults.

Recommendation: drop `SCORE_DELTA_COEF` default from 1.5 → 0.0 and keep
`MARGIN_DELTA_COEF=1.0`. Margin-only shaping is the simpler, better-
performing baseline. (Code default left unchanged for now — user-owned
training artifacts depend on the existing scale; flip after a multi-seed
re-confirmation in a future session.)

### 11.4 Symmetry augmentation — DONE

`python/train.py` augments every collected transition with a column-axis-
reflected copy: board flipped along W, leader/opp_leader col coords
mirrored, action remapped via a precomputed `MIRROR_PERM` (involutive),
mask = `mask[MIRROR_PERM]`. log_prob/value/return/advantage/expert_action
are reused as if on-policy; PPO clipping bounds the policy-mismatch bias.

Caveat: the canonical T&E river layout (`src/engine/types.ts:RIVER_POSITIONS`)
is NOT exactly column-symmetric, so the mirrored sample is a *virtual*
"mirror-world" T&E with a different river. The policy reads the river as
an input channel, so this acts as regularization-via-augmentation rather
than a true game equivariance. Empirical net effect on training is what
matters — not literal symmetry.

Touch points:
- `_compute_mirror_index(idx)` — flat→flat permutation; spatial action
  types (placeTile, placeLeader, placeCatastrophe) flip the col coord;
  others identity.
- `_MIRROR_PERM_NP` — precomputed full permutation, asserted involutive
  at import.
- `_mirror_obs_action(obs, action, mask)` — mirrors board, leaders,
  opp_leaders, action, and mask. Other obs fields pass through.
- `collect_rollout_vec` — after per-env GAE concatenation, optionally
  appends a mirrored copy of every transition. Toggled by
  `SYMMETRY_AUG=1` (default on); set `SYMMETRY_AUG=0` to disable.
  Doubles PPO batch size and ~2× compute per update step.

Mirror invariants verified:
- `MIRROR_PERM[MIRROR_PERM] == identity` (assertion at import).
- Mirrored mask contains the mirrored action: by construction
  `mirrored_mask[mirrored_action] = orig_mask[MIRROR_PERM[MIRROR_PERM[a]]]
  = orig_mask[a] = 1`. The PPO mask-audit assertion (Phase 6.4) does not
  trip on augmented batches.

Verify (TIME_BUDGET=20s, NUM_ENVS=4, SYMMETRY_AUG=1, REPLAY_K=0):
| Metric              | SYMMETRY_AUG=0 (11.1) | SYMMETRY_AUG=1 |
|---------------------|------------------------|-----------------|
| Steps/sec (engine)  | ~292                   | ~235            |
| Mask audit          | passes                 | passes          |
| pg_loss             | -0.037 → -0.022        | -0.054 → 0.020  |
| vf_loss             | 21 → 40                | 0.6 → 38 → 1.6  |
| entropy H           | 4.36 → 3.84            | 4.26 → 3.98     |
| vs_pool_win_rate    | 0.583                  | 0.429 (1 seed)  |

Single-seed result is noisy and below the 11.1 baseline — possibly
because the mirror-world river penalizes the policy's blue-tile placement
priors. Needs a multi-seed (or longer-budget) head-to-head before
recommending the default. `SYMMETRY_AUG` is on by default; flip to 0 if
results don't pan out under the next ablation sweep.

### 11.5 First-class headless tournaments — DONE

`npm run headless` now accepts trained ONNX policies as players, runs
inference via `onnxruntime-node`, and plays them against `simpleAI` (or
each other) without spinning up Python. Same encoder + hierarchical-
argmax decode as the browser path so a single model file works in both
contexts.

Touch points:
- `npm install onnxruntime-node` — added to dependencies. Browser bundle
  size unchanged: onnxruntime-node is only imported from `src/headless/*`,
  which Vite never includes in the production bundle.
- `src/ai/onnxPolicyCore.ts` — new shared module: `buildFeedsForOnnx`,
  `pickActionFromOnnxResult`. Runtime-agnostic via `onnxruntime-common`'s
  `Tensor` type; the runtime-specific adapter passes its own Tensor class.
- `src/ai/onnxPolicy.ts` — slimmed to a thin wrapper over the core +
  onnxruntime-web session loader.
- `src/headless/onnxAdapter.ts` — node-side wrapper using onnxruntime-node;
  caches the loaded session keyed by model path.
- `src/headless/runGame.ts` — `runGame` and `runTournament` now async to
  accommodate `session.run()`. New `aiKinds: ('simple'|'onnx')[]` and
  `onnxModelPath` options. Heuristic path stays sync inside an awaitable
  closure. ONNX failures fall back to `simpleAI` with a one-time warning.
- `src/headless/cli.ts` — `--ai-kind=onnx` (sets seat 0 to ONNX),
  `--ai-kinds=onnx,simple,simple` (per-seat), `--model=<path>` (override
  default `models/policy.onnx`). Wrapped CLI body in an async `main()`.

Verify:
- `npm run headless -- --players=2 --games=10 --ai-kind=onnx` produces a
  coherent leaderboard. Inference runs once per active turn; games end via
  `gameOver` (not `maxActions`), confirming the ONNX policy returns valid
  actions for every turn phase.
- `npm run headless -- --players=2 --games=3` (heuristic-only) regression
  check: still works, results comparable to pre-Phase-11.5 baseline.
- 177 engine tests pass; `npx vite build` succeeds with **unchanged**
  browser bundle (onnxruntime-node is not pulled into the browser).
- `npm run lint` baseline 11 errors unchanged.

The current `models/policy.onnx` reflects a freshly-randomized type_head
(post-Phase-11.1 architecture rewrite, ~20-90s of training) and so loses
0/10 vs simpleAI — the >50% criterion needs a longer training run with a
warm BC start before it's meaningful. The infrastructure is in place; the
metric is a function of training time, not the headless wiring.

---

## Build / verify after each phase

- `npm run test` — engine + AI unit tests
- `npm run build` — TS check + production build
- `npm run lint` — ESLint
- `npm run headless -- --players=2 --games=20` — sanity vs AI matches
