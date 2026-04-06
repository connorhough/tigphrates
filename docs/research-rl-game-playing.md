# Research: RL Methods for Training Game-Playing AI

## Game Characteristics (Tigris & Euphrates)

Before selecting an algorithm, it's critical to classify the game:

| Property | Value |
|---|---|
| Players | 2–4, competitive |
| Turn structure | Turn-based, 2 actions/turn + sub-phases (conflict support, monument choice, war order) |
| Information | **Imperfect** — hands are hidden, bag draws are stochastic |
| Action space | **Large, variable, structured** — union of tile placements (~100+ positions × 4 colors), leader placements, withdrawals, catastrophes, swaps, conflict support commits, pass |
| State space | 11×16 board with multi-layer cell state + per-player hidden hands + bag composition |
| Reward signal | **Sparse & delayed** — final score = minimum across 4 VP colors (unique balancing objective) |
| Game length | Typically 50–150 turns (~100–500 actions total) |
| Stochasticity | Tile draws from bag |

These properties—imperfect information, stochastic elements, large structured action space, delayed/sparse rewards, and multi-player competition—make this a challenging RL problem that rules out naive application of perfect-information algorithms.

---

## Tier 1: Most Promising Approaches

### 1. PPO with Self-Play (Recommended Starting Point)

**What it is:** Proximal Policy Optimization (PPO) is a policy-gradient method that's become the workhorse of modern RL. Combined with self-play (agents train against copies of themselves), it's the most practical starting point.

**Why it fits:**
- Handles large, variable action spaces well via action masking (invalid actions get -∞ logits)
- Works with imperfect information natively — the policy only sees the agent's observation, not full state
- Self-play provides a curriculum of increasingly strong opponents
- Battle-tested on tabletop games via PyTAG framework

**Architecture:**
```
Observation → Encoder (CNN for board + MLP for hand/scores) → Shared trunk → Policy head (action logits)
                                                                            → Value head (scalar)
```

**Key design decisions:**
- **Observation encoding:** Encode the 11×16 board as multi-channel image (channels for: tile colors, leaders by dynasty, monuments, catastrophes, treasures, terrain). Encode hand as count vector. Concatenate scores, treasures, bag-size, turn phase.
- **Action masking:** Use the existing `getValidTilePlacements` / `getValidLeaderPlacements` to mask invalid actions. This is critical for convergence.
- **Reward shaping:** Raw reward (min score at game end) is too sparse. Consider intermediate shaping: +small reward for scoring any VP, +bonus for balancing scores across colors, -penalty for losing conflicts.

**Frameworks:** [CleanRL](https://github.com/vwxyzjn/cleanrl), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [PettingZoo](https://pettingzoo.farama.org/) (multi-agent env wrapper)

**Estimated effort:** Medium. Requires wrapping the game engine as a PettingZoo/Gymnasium environment.

**References:**
- PyTAG used PPO+self-play successfully on Stratego, Catan, and other tabletop games
- The PPO paper: Schulman et al., 2017

---

### 2. MuZero / Stochastic MuZero (via LightZero)

**What it is:** MuZero learns a dynamics model of the environment and uses MCTS for planning at decision time — without needing the actual game rules for search. Stochastic MuZero extends this to handle random events (like tile draws).

**Why it fits:**
- Model-based planning is powerful for games with long-horizon consequences (monuments, kingdom building)
- Stochastic MuZero handles the random tile draws
- MCTS naturally handles the large action space by focusing search on promising moves
- LightZero provides a production-ready implementation

**Challenges:**
- MuZero assumes perfect information by default. For imperfect information (hidden hands), need adaptations:
  - **Information Set MCTS:** Average over possible opponent hands
  - **Belief state modeling:** Add auxiliary heads to predict hidden state
  - **Perfect Information Monte Carlo (PIMC):** Sample concrete states from beliefs, run MCTS on each, aggregate (simple but effective baseline — see AlphaZe** below)
- Higher compute cost than PPO — MCTS runs many simulations per move
- More complex to implement and tune

**Framework:** [LightZero](https://github.com/opendilab/LightZero) (NeurIPS 2023 Spotlight) — unified benchmark with MuZero, EfficientZero, Stochastic MuZero, and Sampled MuZero all implemented.

**Estimated effort:** High. Need to define observation/action spaces for LightZero, handle imperfect information.

---

### 3. AlphaZe** (AlphaZero + PIMC for Imperfect Information)

**What it is:** A surprisingly strong baseline that adapts AlphaZero to imperfect information games by replacing standard MCTS with Perfect Information Monte Carlo (PIMC) search. At decision time, sample multiple "determinizations" (fill in hidden info randomly), run MCTS on each, and combine results.

**Why it fits:**
- Directly addresses the imperfect information challenge
- Simpler than full game-theoretic approaches (ReBeL, Student of Games)
- Empirically competitive — the original paper showed it matches or beats more complex approaches on several games
- Can reuse the existing game engine's `gameReducer` for forward simulation

**How it works:**
1. Train a neural network (policy + value) via self-play on determinized states
2. At decision time: sample N possible opponent hands consistent with observations
3. Run MCTS on each determinization
4. Aggregate action recommendations across all determinizations
5. Pick the most-recommended action

**Reference:** Schmid et al., "AlphaZe**: AlphaZero-like baselines for imperfect information games are surprisingly strong" (Frontiers in AI, 2023)

---

## Tier 2: Game-Theoretic Approaches (More Powerful, More Complex)

### 4. ReBeL (Recursive Belief-based Learning)

**What it is:** Meta's framework that combines RL self-play with search in imperfect information games. Converges to Nash equilibrium in 2-player zero-sum games. Achieved superhuman poker play.

**Why it fits:** Handles imperfect information with theoretical guarantees. Reduces to AlphaZero-like algorithm in perfect information case.

**Challenges:**
- Designed for 2-player zero-sum — T&E is 2–4 player and not strictly zero-sum
- Belief tracking over the "public belief state" is complex for T&E's state space
- No off-the-shelf implementation for custom games

**Reference:** Brown et al., "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" (NeurIPS 2020)

### 5. Student of Games (SoG)

**What it is:** The most general algorithm — unifies guided search, self-play learning, and game-theoretic reasoning. Works on both perfect and imperfect information games with theoretical convergence guarantees.

**Why it fits:** Most theoretically complete approach. Handles imperfect information, multiple players, and stochastic elements.

**Challenges:**
- Very complex to implement from scratch
- High computational requirements
- Limited open-source implementations

**Reference:** Schmid et al., "Student of Games" (Science Advances, 2023)

### 6. DeepNash

**What it is:** Model-free multiagent RL that achieved human expert-level play in Stratego (a large imperfect information game). Uses Regularized Nash Dynamics (R-NaD) for policy optimization.

**Why it fits:** Stratego shares key properties with T&E: imperfect information, large state/action space, strategic depth. Model-free, so no need to learn a dynamics model.

**Challenges:** Designed for 2-player zero-sum. Significant compute requirements.

**Reference:** Perolat et al., "Mastering the game of Stratego with model-free multiagent reinforcement learning" (Science, 2022)

---

## Tier 3: Simpler Baselines (For Comparison & Incremental Improvement)

### 7. Enhanced Heuristic / Rule-Based (Current simpleAI)

The existing `simpleAI.ts` is a reasonable heuristic baseline. Improvements:
- Better leader placement (consider kingdom size, conflict risk)
- Catastrophe usage (currently unused)
- Smarter tile placement (consider monument building, blocking opponents)
- Score-balancing heuristic aligned with min-score objective

### 8. Monte Carlo Tree Search (Pure, No Neural Network)

MCTS with random rollouts. Cheap to implement, provides a stronger baseline than heuristics. Can use PIMC for imperfect information handling.

### 9. DQN / Rainbow

Deep Q-Learning variants. Less suitable here due to the large, structured action space (would need action decomposition), but mentioned for completeness.

---

## Recommended Implementation Roadmap

### Phase 1: Environment & Baselines
1. **Wrap the game engine as a Gymnasium/PettingZoo environment** — expose observation tensors and action spaces from the TypeScript engine (either via a Python binding or rewrite core logic in Python)
2. **Define observation encoding** — multi-channel board tensor + hand/score vectors
3. **Define action encoding** — flat action space with masking, or hierarchical (action type → parameters)
4. **Benchmark the existing simpleAI** — run tournaments to establish baseline win rates

### Phase 2: PPO + Self-Play
5. **Train PPO agent** against copies of itself using CleanRL or SB3
6. **Iterate on reward shaping** — start with sparse (game outcome only), add shaping if needed
7. **Evaluate against simpleAI** — target: >70% win rate

### Phase 3: Search-Based Methods
8. **Add MCTS** — pure MCTS with PIMC for imperfect info
9. **AlphaZe\*\*-style training** — neural network + PIMC-MCTS self-play
10. **Optionally explore MuZero via LightZero** for model-based planning

### Phase 4: Advanced (if needed)
11. Population-based training (train a league of diverse agents)
12. Game-theoretic methods (ReBeL-inspired) for Nash convergence
13. Curriculum learning (start with simplified 2-player, scale to 3-4)

---

## Key Technical Considerations

### Language Bridge
The game engine is in TypeScript. Options:
- **Option A:** Call the TS engine from Python via subprocess/WASM — preserves existing tested logic
- **Option B:** Port engine to Python — easier RL integration but risks divergence
- **Option C:** Train in JS/TS using TensorFlow.js or ONNX Runtime — avoids bridge entirely

### Observation Space Design
```
Board channels (11 × 16 × N):
  - 4 channels: tile presence by color (binary)
  - 4 channels: leader presence by color (one-hot per dynasty)
  - 1 channel: monuments
  - 1 channel: catastrophes
  - 1 channel: treasures
  - 1 channel: terrain (river vs land)
  - 1 channel: flipped tiles

Player-specific vector:
  - Hand tile counts (4 values)
  - Scores per color (4 values)
  - Treasure count
  - Catastrophes remaining
  - Leader positions (4 × 2 values, or -1 if off-board)
  - Bag size
  - Actions remaining
  - Turn phase (one-hot)
```

### Action Space Design
Flatten all possible actions into a single discrete space with masking:

| Action type | Parameters | Approx count |
|---|---|---|
| placeTile | 4 colors × 176 positions | ~704 |
| placeLeader | 4 colors × 176 positions | ~704 |
| withdrawLeader | 4 colors | 4 |
| placeCatastrophe | 176 positions | 176 |
| swapTiles | 2^6 subsets | 64 |
| commitSupport | 2^6 subsets | 64 |
| chooseWarOrder | 4 colors | 4 |
| buildMonument | 6 IDs | 6 |
| declineMonument | — | 1 |
| pass | — | 1 |
| **Total** | | **~1,728** |

With action masking, only valid actions get non-zero probability. This is manageable for PPO.

### Reward Design
```
Terminal reward: +1 for win, -1 for loss, 0 for draw (simplest)

Optional shaping:
  - min(scores) / max_possible_score  (normalized min-score)
  - Δ min(scores) per turn            (incremental progress)
  - bonus for balanced scoring         (entropy across color scores)
```

---

## Key Libraries & Frameworks

| Library | Use Case | URL |
|---|---|---|
| **PettingZoo** | Multi-agent env API | https://pettingzoo.farama.org/ |
| **CleanRL** | Single-file PPO implementation | https://github.com/vwxyzjn/cleanrl |
| **Stable-Baselines3** | Production RL algorithms | https://github.com/DLR-RM/stable-baselines3 |
| **LightZero** | MuZero/AlphaZero family | https://github.com/opendilab/LightZero |
| **OpenSpiel** | Game-theoretic algorithms | https://github.com/google-deepmind/open_spiel |
| **PyTAG** | Tabletop game RL benchmark | https://github.com/GAIGResearch/PyTAG |
| **TensorFlow.js** | Training in JS (no bridge) | https://www.tensorflow.org/js |

---

## Sources

- [PyTAG: Tabletop Games for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2405.18123)
- [AlphaZe**: AlphaZero-like baselines for imperfect information games are surprisingly strong](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full)
- [Student of Games: A unified learning algorithm](https://www.science.org/doi/10.1126/sciadv.adg3256)
- [ReBeL: Combining Deep RL and Search for Imperfect-Information Games](https://arxiv.org/abs/2007.13544)
- [Mastering Stratego with model-free multiagent RL (DeepNash)](https://www.science.org/doi/10.1126/science.add4679)
- [LightZero: Unified Benchmark for MCTS](https://github.com/opendilab/LightZero)
- [Deep RL from Self-Play in Imperfect-Information Games](https://arxiv.org/pdf/1603.01121)
- [RL Agents Playing Ticket to Ride](https://ieeexplore.ieee.org/document/10154465/)
- [Self-Play Meta-RL in Multi-Agent Games](https://link.springer.com/article/10.1007/s44427-026-00021-y)
- [Evolutionary RL with Action Sequence Search for Imperfect Information Games](https://www.sciencedirect.com/science/article/abs/pii/S0020025524007187)
