# CLAUDE.md

## Project overview

Tigphrates is a digital implementation of Tigris & Euphrates (Reiner Knizia, 1997), a 2-4 player strategic board game set in ancient Mesopotamia. It includes a React web UI, a pure TypeScript game engine, a heuristic AI, and a Python RL training pipeline.

## Commands

```bash
npm run dev          # Start Vite dev server
npm run build        # TypeScript check + Vite production build
npm run test         # Run all vitest tests
npm run lint         # ESLint
npm run headless     # Run AI vs AI games: --players=N --games=N --turns=N --log --no-log
npm run bridge       # Start the JSON-RPC bridge server (for Python RL training)
```

Python RL training (from repo root):
```bash
pip install -r python/requirements.txt  # numpy, gymnasium, torch
python python/train.py                  # 5-min PPO training + eval vs heuristic AI
./python/run_experiments.sh             # Automated experiment loop
```

## Architecture

### Game engine (`src/engine/`)
Pure TypeScript, no UI dependencies. Functional style — `gameReducer(state, action)` returns new state.
- `types.ts` — All types: `GameState`, `GameAction`, `Cell`, `Player`, `TurnPhase`, board constants (11×16 grid)
- `reducer.ts` — Main entry point. Validates phase/player, delegates to `actions.ts`, auto-ends turns
- `actions.ts` — Action handlers: placeTile, placeLeader, placeCatastrophe, swapTiles, commitSupport, chooseWarOrder, buildMonument
- `board.ts` — Grid operations: `findKingdoms`, `findConnectedGroup`, `getNeighbors`
- `validation.ts` — `getValidTilePlacements`, `getValidLeaderPlacements`, `canPlaceCatastrophe`
- `conflict.ts` — Revolt and war resolution
- `monument.ts` — Monument building detection and placement
- `turn.ts` — End-of-turn: draw tiles, monument VP scoring, treasure collection, game-end check
- `setup.ts` — `createGame(playerCount, aiFlags)` — initial state factory

### UI (`src/components/`)
React components rendering the board, hand, dialogs for conflicts/monuments/war-order.

### AI (`src/ai/`)
- `simpleAI.ts` — Heuristic agent. Handles all turn phases. Prioritizes: place leaders → score tiles → place any tile → swap. Greedy conflict support. Never uses catastrophes.

### Headless runner (`src/headless/`)
- `runGame.ts` — `runGame(opts)` and `runTournament(gameCount, playerCount)` for AI vs AI games with compact logging
- `cli.ts` — CLI entry point for `npm run headless`

### Python bridge (`src/bridge/`)
- `server.ts` — JSON-RPC server over stdin/stdout. Methods: `create`, `step`, `valid_actions`, `get_observation`, `ai_action`, `reset`. Encodes board as 13-channel tensor, flat 1728-action space with masking.

### Python RL (`python/`)
- `tigphrates_env.py` — Gymnasium `TigphratesEnv` (single-agent vs heuristic) and `TigphratesMultiAgentEnv` (all-agent self-play). Spawns Node subprocess for bridge.
- `train.py` — PPO training script. CNN board encoder + MLP actor-critic. Designed for autonomous modification via autoresearch pattern.
- `evaluate.py` — Evaluation harness. Plays N games vs heuristic, reports win_rate/avg_min_score/avg_margin.
- `program.md` — Instructions for autonomous RL experiment loop.
- `run_experiments.sh` — Script-driven experiment loop. Calls Claude stateless for each edit.

## Key patterns

- Game state is immutable-by-convention; `actions.ts` uses `structuredClone` internally
- Turn phases: `action` → (optional) `conflictSupport` / `warOrderChoice` / `monumentChoice` → back to `action` or `gameOver`
- Players get 2 actions per turn, then draw/score/collect/check-game-end
- The `activePlayer` during `conflictSupport` depends on who hasn't committed yet (attacker first, then defender)
- Scoring: final score = minimum across 4 VP colors + treasure wildcards

## Testing

Tests are in `__tests__/` directories next to source files. Run `npm run test`. Engine tests cover: actions, board, catastrophe, monument, revolt, war, setup, turn, types, validation, reducer, and simpleAI.

## Game rules

See `GAME_RULES.md` for the complete rules reference. Key points:
- 11×16 board with river/land terrain. Blue tiles go on river, others on land.
- 4 leader colors (red priest, blue farmer, green trader, black king). King is wildcard scorer.
- Conflicts: revolts (internal, red tiles only) and wars (external, matching color tiles).
- Monuments: 2×2 same-color tiles → permanent scoring structure.
- Game ends when ≤2 treasures remain or bag can't refill a hand.
