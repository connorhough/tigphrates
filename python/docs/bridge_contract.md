# Bridge Contract (auto-generated)

> Source of truth: `src/bridge/encoder.ts`. Do not edit by hand.
> Regenerate via `npm run bridge:dump-contract`.
> The TS bridge is frozen for the duration of any RL run — Python-side edits
> only.

## Constants

| name | value |
|---|---|
| BOARD_ROWS | 11 |
| BOARD_COLS | 16 |
| BOARD_CHANNELS | 15 |
| HAND_MAX | 6 |
| ACTION_SPACE_SIZE | 1728 |
| color order | red, blue, green, black |

## Action space (flat, hierarchical layout)

Sample type first (10-way Cat), then param conditional on type.
Per-state mask sums collapse from up to 1728 to 10
types + at most 704 params (usually <50 in either head).

| typeIdx | name | flat range | size | param semantics |
|---|---|---|---|---|
| 0 | `placeTile` | 0–703 | 704 | colorIdx*176 + row*16 + col |
| 1 | `placeLeader` | 704–1407 | 704 | colorIdx*176 + row*16 + col |
| 2 | `withdrawLeader` | 1408–1411 | 4 | colorIdx (0=red,1=blue,2=green,3=black) |
| 3 | `placeCatastrophe` | 1412–1587 | 176 | row*16 + col |
| 4 | `swapTiles` | 1588–1651 | 64 | 6-bit hand mask (bit i = discard hand[i]) |
| 5 | `pass` | 1652 | 1 | singleton (paramIdx=0) |
| 6 | `commitSupport` | 1653–1716 | 64 | 6-bit hand mask of tiles to commit |
| 7 | `chooseWarOrder` | 1717–1720 | 4 | colorIdx (which conflict to resolve first) |
| 8 | `buildMonument` | 1721–1726 | 6 | monument-template index 0–5 |
| 9 | `declineMonument` | 1727 | 1 | singleton (paramIdx=0) |

## Observation: board tensor (shape 15×11×16)

- 0–3:  tile color (binary, COLOR_INDEX order: red, blue, green, black)
- 4–7:  leader color (value = 1 + ownerPlayerIndex, 0 = empty)
- 8:    monument color1 = 1 + COLOR_INDEX, 0 = none
- 9:    monument color2 = 1 + COLOR_INDEX
- 10:   monument owner = 1 + ownerPlayerIndex
- 11:   catastrophe (binary)
- 12:   treasure (binary)
- 13:   river terrain (binary)
- 14:   tileFlipped (binary)

## Observation: scalar / vector fields

- `hand` int[4]: count by color
- `handSeq` int[6]: ordered hand, -1 = empty slot
- `scores` int[4], `treasures` int, `catastrophesRemaining` int
- `leaderPositions` int[8]: (row,col) pairs for own 4 leaders, -1,-1 = unplaced
- `opponentScores` int[N-1][4], `opponentLeaderPositions` int[N-1][8]
- `bagSize` int, `actionsRemaining` int
- `turnPhase`: 0=action, 1=conflictSupport, 2=warOrderChoice, 3=monumentChoice, 4=gameOver
- `currentPlayer` int, `playerIndex` int (perspective), `numPlayers` int
- `conflict` (nullable):
  - `type`: 0=revolt, 1=war
  - `color`: COLOR_INDEX of the contested color
  - `attackerStrength`, `defenderStrength`: int (committed support tally)
  - `attackerCommitted`: bool (true once attacker has chosen support)
  - `isAttacker`, `isDefender`: bool from this player's perspective
  - `attackerHandSize`, `defenderHandSize`: int (max additional support possible)

## Reward (terminal-only on the TS side)

- `gameOver`: +1 to winner (max of min-color-score + treasures), -1 to others.
- All intermediate shaping is computed in Python from observation deltas.
  Do not add intermediate signal in the bridge.

## RPCs (Python → TS)

`create`, `reset`, `step`, `step_action`, `get_state`, `get_observation`,
`valid_actions`, `ai_action`, `load_state`, `decode_action`, `delete_game`,
`agent_step` (fast path: applies agent action + drives heuristic opponents,
returns next obs + mask in one round-trip).
