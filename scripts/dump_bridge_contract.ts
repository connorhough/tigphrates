/**
 * Emit a distilled bridge contract for Python-side consumers (RL prompts,
 * humans, autoresearch loop). Imports authoritative constants from
 * src/bridge/encoder.ts so the document cannot drift from the runtime.
 *
 * Run: npm run bridge:dump-contract
 *      (writes to python/docs/bridge_contract.md)
 *
 * The script fails loudly if invariants between encoder constants and the
 * hand-curated PARAM_DOCS table go out of sync — that is the rot detector.
 */
import {
  ACTION_SPACE_SIZE,
  ACTION_TYPES,
  TYPE_PARAM_SIZES,
  TYPE_BASES,
  BOARD_CHANNELS,
  HAND_MAX,
  COLOR_INDEX,
} from '../src/bridge/encoder'
import { BOARD_ROWS, BOARD_COLS } from '../src/engine/types'

// Param semantics — must mirror enumerateValidActions in encoder.ts.
// Keep this map and ACTION_TYPES in lockstep; the drift check below
// fails the build if a new type is added without a doc entry.
const PARAM_DOCS: Record<string, string> = {
  placeTile:        `colorIdx*${BOARD_ROWS * BOARD_COLS} + row*${BOARD_COLS} + col`,
  placeLeader:      `colorIdx*${BOARD_ROWS * BOARD_COLS} + row*${BOARD_COLS} + col`,
  withdrawLeader:   'colorIdx (0=red,1=blue,2=green,3=black)',
  placeCatastrophe: `row*${BOARD_COLS} + col`,
  swapTiles:        '6-bit hand mask (bit i = discard hand[i])',
  pass:             'singleton (paramIdx=0)',
  commitSupport:    '6-bit hand mask of tiles to commit',
  chooseWarOrder:   'colorIdx (which conflict to resolve first)',
  buildMonument:    'monument-template index 0–5',
  declineMonument:  'singleton (paramIdx=0)',
}

// --- Drift checks ---

if (TYPE_PARAM_SIZES.length !== ACTION_TYPES.length) {
  throw new Error(
    `ACTION_TYPES (${ACTION_TYPES.length}) and TYPE_PARAM_SIZES (${TYPE_PARAM_SIZES.length}) length mismatch`,
  )
}
const sumOfTypes = TYPE_PARAM_SIZES.reduce((a, b) => a + b, 0)
if (sumOfTypes !== ACTION_SPACE_SIZE) {
  throw new Error(
    `ACTION_SPACE_SIZE (${ACTION_SPACE_SIZE}) != sum(TYPE_PARAM_SIZES) (${sumOfTypes})`,
  )
}
for (const t of ACTION_TYPES) {
  if (!(t in PARAM_DOCS)) {
    throw new Error(`PARAM_DOCS missing entry for action type "${t}"`)
  }
}
for (const k of Object.keys(PARAM_DOCS)) {
  if (!ACTION_TYPES.includes(k as typeof ACTION_TYPES[number])) {
    throw new Error(`PARAM_DOCS has stale entry "${k}" (not in ACTION_TYPES)`)
  }
}

// --- Render ---

const colorOrder = Object.entries(COLOR_INDEX)
  .sort((a, b) => a[1] - b[1])
  .map(([k]) => k)
  .join(', ')

const actionRows = ACTION_TYPES.map((name, i) => {
  const base = TYPE_BASES[i]
  const size = TYPE_PARAM_SIZES[i]
  const end = base + size - 1
  const range = size === 1 ? `${base}` : `${base}–${end}`
  return `| ${i} | \`${name}\` | ${range} | ${size} | ${PARAM_DOCS[name]} |`
}).join('\n')

const obsChannels = [
  '0–3:  tile color (binary, COLOR_INDEX order: red, blue, green, black)',
  '4–7:  leader color (value = 1 + ownerPlayerIndex, 0 = empty)',
  '8:    monument color1 = 1 + COLOR_INDEX, 0 = none',
  '9:    monument color2 = 1 + COLOR_INDEX',
  '10:   monument owner = 1 + ownerPlayerIndex',
  '11:   catastrophe (binary)',
  '12:   treasure (binary)',
  '13:   river terrain (binary)',
  '14:   tileFlipped (binary)',
].map((s) => '- ' + s).join('\n')

const out = `# Bridge Contract (auto-generated)

> Source of truth: \`src/bridge/encoder.ts\`. Do not edit by hand.
> Regenerate via \`npm run bridge:dump-contract\`.
> The TS bridge is frozen for the duration of any RL run — Python-side edits
> only.

## Constants

| name | value |
|---|---|
| BOARD_ROWS | ${BOARD_ROWS} |
| BOARD_COLS | ${BOARD_COLS} |
| BOARD_CHANNELS | ${BOARD_CHANNELS} |
| HAND_MAX | ${HAND_MAX} |
| ACTION_SPACE_SIZE | ${ACTION_SPACE_SIZE} |
| color order | ${colorOrder} |

## Action space (flat, hierarchical layout)

Sample type first (${ACTION_TYPES.length}-way Cat), then param conditional on type.
Per-state mask sums collapse from up to ${ACTION_SPACE_SIZE} to ${ACTION_TYPES.length}
types + at most ${Math.max(...TYPE_PARAM_SIZES)} params (usually <50 in either head).

| typeIdx | name | flat range | size | param semantics |
|---|---|---|---|---|
${actionRows}

## Observation: board tensor (shape ${BOARD_CHANNELS}×${BOARD_ROWS}×${BOARD_COLS})

${obsChannels}

## Observation: scalar / vector fields

- \`hand\` int[4]: count by color
- \`handSeq\` int[${HAND_MAX}]: ordered hand, -1 = empty slot
- \`scores\` int[4], \`treasures\` int, \`catastrophesRemaining\` int
- \`leaderPositions\` int[8]: (row,col) pairs for own 4 leaders, -1,-1 = unplaced
- \`opponentScores\` int[N-1][4], \`opponentLeaderPositions\` int[N-1][8]
- \`bagSize\` int, \`actionsRemaining\` int
- \`turnPhase\`: 0=action, 1=conflictSupport, 2=warOrderChoice, 3=monumentChoice, 4=gameOver
- \`currentPlayer\` int, \`playerIndex\` int (perspective), \`numPlayers\` int
- \`conflict\` (nullable):
  - \`type\`: 0=revolt, 1=war
  - \`color\`: COLOR_INDEX of the contested color
  - \`attackerStrength\`, \`defenderStrength\`: int (committed support tally)
  - \`attackerCommitted\`: bool (true once attacker has chosen support)
  - \`isAttacker\`, \`isDefender\`: bool from this player's perspective
  - \`attackerHandSize\`, \`defenderHandSize\`: int (max additional support possible)

## Reward (terminal-only on the TS side)

- \`gameOver\`: +1 to winner (max of min-color-score + treasures), -1 to others.
- All intermediate shaping is computed in Python from observation deltas.
  Do not add intermediate signal in the bridge.

## RPCs (Python → TS)

\`create\`, \`reset\`, \`step\`, \`step_action\`, \`get_state\`, \`get_observation\`,
\`valid_actions\`, \`ai_action\`, \`load_state\`, \`decode_action\`, \`delete_game\`,
\`agent_step\` (fast path: applies agent action + drives heuristic opponents,
returns next obs + mask in one round-trip).
`

process.stdout.write(out)
