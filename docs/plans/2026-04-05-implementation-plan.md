# Tigris & Euphrates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a playable web version of Tigris & Euphrates with AI opponents.

**Architecture:** Pure TypeScript game engine (state machine) with React + Canvas 2D frontend. Engine is fully decoupled from UI -- takes state + action, returns new state. React useReducer wraps the engine. Canvas renders the board, DOM renders UI chrome.

**Tech Stack:** Vite, React 18, TypeScript, Canvas 2D, Vitest for testing.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`
- Create: `src/main.tsx`, `src/App.tsx`
- Create: `vitest.config.ts`

**Step 1: Scaffold Vite project**

Run:
```bash
cd /Users/connor/Src/tigphrates
npm create vite@latest . -- --template react-ts
```

If the directory is not empty, accept overwrite prompts.

**Step 2: Install dependencies**

```bash
npm install
npm install -D vitest
```

**Step 3: Configure Vitest**

Add to `vite.config.ts`:
```typescript
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'node',
  },
})
```

Add to `tsconfig.json` compilerOptions: `"types": ["vitest/globals"]`

Add to `package.json` scripts: `"test": "vitest run", "test:watch": "vitest"`

**Step 4: Verify it works**

```bash
npm run build
npm test
```

**Step 5: Initialize git and commit**

```bash
git init
git add -A
git commit -m "scaffold vite + react + typescript project with vitest"
```

---

### Task 2: Core Types

**Files:**
- Create: `src/engine/types.ts`
- Test: `src/engine/__tests__/types.test.ts`

**Step 1: Write type smoke test**

```typescript
// src/engine/__tests__/types.test.ts
import { createInitialBoard, BOARD_ROWS, BOARD_COLS } from '../types'

describe('types', () => {
  it('creates a board with correct dimensions', () => {
    const board = createInitialBoard()
    expect(board).toHaveLength(BOARD_ROWS)
    expect(board[0]).toHaveLength(BOARD_COLS)
  })

  it('marks river cells correctly', () => {
    const board = createInitialBoard()
    // Row 0, col 4 should be river
    expect(board[0][4].terrain).toBe('river')
    // Row 0, col 0 should be land
    expect(board[0][0].terrain).toBe('land')
  })

  it('places starting temples with treasures', () => {
    const board = createInitialBoard()
    // (col 10, row 0) is a starting temple
    expect(board[0][10].tile).toBe('red')
    expect(board[0][10].hasTreasure).toBe(true)
    // (col 0, row 0) is empty land
    expect(board[0][0].tile).toBeNull()
    expect(board[0][0].hasTreasure).toBe(false)
  })
})
```

**Step 2: Run test to verify it fails**

```bash
npx vitest run src/engine/__tests__/types.test.ts
```

**Step 3: Implement types and board factory**

```typescript
// src/engine/types.ts

export const BOARD_ROWS = 11
export const BOARD_COLS = 16

export type TileColor = 'red' | 'blue' | 'green' | 'black'
export type LeaderColor = 'red' | 'blue' | 'green' | 'black'
export type Dynasty = 'archer' | 'bull' | 'pot' | 'lion'
export type Terrain = 'land' | 'river'

export interface Position {
  row: number
  col: number
}

export interface Leader {
  color: LeaderColor
  dynasty: Dynasty
}

export interface Cell {
  terrain: Terrain
  tile: TileColor | null
  tileFlipped: boolean
  leader: Leader | null
  catastrophe: boolean
  monument: MonumentId | null
  hasTreasure: boolean
}

export type MonumentId = string // e.g. "red-blue", "green-black"

export interface Monument {
  id: MonumentId
  color1: TileColor
  color2: TileColor
  position: Position | null // top-left of 2x2, null if unplaced
}

export type TurnPhase =
  | 'action'
  | 'conflictSupport'   // waiting for attacker/defender to commit tiles
  | 'warOrderChoice'    // active player choosing which war to resolve
  | 'monumentChoice'    // active player choosing whether to build monument
  | 'gameOver'

export interface ConflictState {
  type: 'revolt' | 'war'
  color: LeaderColor
  attacker: { playerIndex: number; position: Position }
  defender: { playerIndex: number; position: Position }
  attackerStrength: number
  defenderStrength: number
  attackerCommitted: number[] | null  // null = hasn't committed yet
  defenderCommitted: number[] | null
  // For war: pending wars still to resolve
  pendingWarColors?: LeaderColor[]
  // For war: the unification tile position
  unificationTilePosition?: Position
}

export interface Player {
  dynasty: Dynasty
  hand: TileColor[]
  leaders: { color: LeaderColor; position: Position | null }[]
  catastrophesRemaining: number
  score: Record<TileColor, number>
  treasures: number
  isAI: boolean
}

export interface GameState {
  board: Cell[][]
  players: Player[]
  bag: TileColor[]
  monuments: Monument[]
  currentPlayer: number
  actionsRemaining: number
  turnPhase: TurnPhase
  pendingConflict: ConflictState | null
  pendingMonument: { position: Position; color: TileColor } | null
}

export type GameAction =
  | { type: 'placeLeader'; color: LeaderColor; position: Position }
  | { type: 'withdrawLeader'; color: LeaderColor }
  | { type: 'placeTile'; color: TileColor; position: Position }
  | { type: 'placeCatastrophe'; position: Position }
  | { type: 'swapTiles'; indices: number[] }
  | { type: 'commitSupport'; indices: number[] }
  | { type: 'chooseWarOrder'; color: LeaderColor }
  | { type: 'buildMonument'; monumentId: MonumentId }
  | { type: 'declineMonument' }
  | { type: 'pass' }

// Board layout: river positions
const RIVER_POSITIONS: [number, number][] = [
  // row, col
  [0,4],[0,5],[0,6],[0,7],[0,8],[0,12],
  [1,4],[1,12],
  [2,3],[2,4],[2,12],[2,13],
  [3,0],[3,1],[3,2],[3,3],[3,13],[3,14],[3,15],
  [4,14],[4,15],
  [5,14],
  [6,0],[6,1],[6,2],[6,3],[6,12],[6,13],[6,14],
  [7,3],[7,4],[7,5],[7,6],[7,7],[7,12],
  [8,7],[8,8],[8,9],[8,10],[8,11],[8,12],
]

const STARTING_TEMPLES: [number, number][] = [
  [0,10],[1,1],[1,15],[2,5],[4,13],[6,9],[7,1],[8,14],[9,6],[10,10],
]

export function createInitialBoard(): Cell[][] {
  const riverSet = new Set(RIVER_POSITIONS.map(([r,c]) => `${r},${c}`))
  const templeSet = new Set(STARTING_TEMPLES.map(([r,c]) => `${r},${c}`))

  const board: Cell[][] = []
  for (let row = 0; row < BOARD_ROWS; row++) {
    const rowCells: Cell[] = []
    for (let col = 0; col < BOARD_COLS; col++) {
      const key = `${row},${col}`
      const isTemple = templeSet.has(key)
      rowCells.push({
        terrain: riverSet.has(key) ? 'river' : 'land',
        tile: isTemple ? 'red' : null,
        tileFlipped: false,
        leader: null,
        catastrophe: false,
        monument: null,
        hasTreasure: isTemple,
      })
    }
    board.push(rowCells)
  }
  return board
}

export const ALL_MONUMENTS: Monument[] = [
  { id: 'red-blue', color1: 'red', color2: 'blue', position: null },
  { id: 'red-green', color1: 'red', color2: 'green', position: null },
  { id: 'red-black', color1: 'red', color2: 'black', position: null },
  { id: 'blue-green', color1: 'blue', color2: 'green', position: null },
  { id: 'blue-black', color1: 'blue', color2: 'black', position: null },
  { id: 'green-black', color1: 'green', color2: 'black', position: null },
]

export const TILE_COUNTS: Record<TileColor, number> = {
  red: 57,
  blue: 36,
  green: 30,
  black: 30,
}
```

**Step 4: Run test to verify it passes**

```bash
npx vitest run src/engine/__tests__/types.test.ts
```

**Step 5: Commit**

```bash
git add src/engine/types.ts src/engine/__tests__/types.test.ts
git commit -m "add core game types and board factory"
```

---

### Task 3: Game Setup (createGame)

**Files:**
- Create: `src/engine/setup.ts`
- Test: `src/engine/__tests__/setup.test.ts`

**Step 1: Write failing tests**

```typescript
// src/engine/__tests__/setup.test.ts
import { createGame } from '../setup'

describe('createGame', () => {
  it('creates a 2-player game with correct initial state', () => {
    const state = createGame(2)
    expect(state.players).toHaveLength(2)
    expect(state.currentPlayer).toBe(0)
    expect(state.actionsRemaining).toBe(2)
    expect(state.turnPhase).toBe('action')
  })

  it('gives each player 6 tiles', () => {
    const state = createGame(2)
    for (const player of state.players) {
      expect(player.hand).toHaveLength(6)
    }
  })

  it('gives each player 4 leaders all off-board', () => {
    const state = createGame(2)
    for (const player of state.players) {
      expect(player.leaders).toHaveLength(4)
      for (const leader of player.leaders) {
        expect(leader.position).toBeNull()
      }
    }
  })

  it('gives each player 2 catastrophe tiles', () => {
    const state = createGame(2)
    for (const player of state.players) {
      expect(player.catastrophesRemaining).toBe(2)
    }
  })

  it('sets up the bag with correct total after dealing', () => {
    const state = createGame(2)
    // 153 total - 10 starting temples - 12 dealt (6 per player) = 131
    expect(state.bag.length + 10 + 12).toBe(153)
  })

  it('initializes all scores to zero', () => {
    const state = createGame(2)
    for (const player of state.players) {
      expect(player.score).toEqual({ red: 0, blue: 0, green: 0, black: 0 })
      expect(player.treasures).toBe(0)
    }
  })

  it('marks second player as AI when specified', () => {
    const state = createGame(2, [false, true])
    expect(state.players[0].isAI).toBe(false)
    expect(state.players[1].isAI).toBe(true)
  })

  it('initializes 6 available monuments', () => {
    const state = createGame(2)
    expect(state.monuments).toHaveLength(6)
    for (const m of state.monuments) {
      expect(m.position).toBeNull()
    }
  })
})
```

**Step 2: Run test to verify failure**

**Step 3: Implement createGame**

```typescript
// src/engine/setup.ts
import {
  GameState, Player, Dynasty, TileColor, LeaderColor,
  TILE_COUNTS, ALL_MONUMENTS, createInitialBoard
} from './types'

const DYNASTIES: Dynasty[] = ['archer', 'bull', 'pot', 'lion']
const LEADER_COLORS: LeaderColor[] = ['red', 'blue', 'green', 'black']

function createBag(): TileColor[] {
  const bag: TileColor[] = []
  for (const [color, count] of Object.entries(TILE_COUNTS)) {
    // Subtract 10 starting temples from red
    const adjusted = color === 'red' ? count - 10 : count
    for (let i = 0; i < adjusted; i++) {
      bag.push(color as TileColor)
    }
  }
  return shuffle(bag)
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function drawTiles(bag: TileColor[], count: number): { drawn: TileColor[]; remaining: TileColor[] } {
  return {
    drawn: bag.slice(0, count),
    remaining: bag.slice(count),
  }
}

export function createGame(
  playerCount: number,
  aiFlags?: boolean[],
): GameState {
  let bag = createBag()
  const players: Player[] = []

  for (let i = 0; i < playerCount; i++) {
    const { drawn, remaining } = drawTiles(bag, 6)
    bag = remaining
    players.push({
      dynasty: DYNASTIES[i],
      hand: drawn,
      leaders: LEADER_COLORS.map(color => ({ color, position: null })),
      catastrophesRemaining: 2,
      score: { red: 0, blue: 0, green: 0, black: 0 },
      treasures: 0,
      isAI: aiFlags?.[i] ?? (i > 0),
    })
  }

  return {
    board: createInitialBoard(),
    players,
    bag,
    monuments: ALL_MONUMENTS.map(m => ({ ...m })),
    currentPlayer: 0,
    actionsRemaining: 2,
    turnPhase: 'action',
    pendingConflict: null,
    pendingMonument: null,
  }
}
```

**Step 4: Run tests to verify pass**

**Step 5: Commit**

```bash
git add src/engine/setup.ts src/engine/__tests__/setup.test.ts
git commit -m "add game setup with player initialization and tile bag"
```

---

### Task 4: Board Utilities (adjacency, flood fill, kingdom detection)

**Files:**
- Create: `src/engine/board.ts`
- Test: `src/engine/__tests__/board.test.ts`

**Step 1: Write failing tests**

```typescript
// src/engine/__tests__/board.test.ts
import { getNeighbors, findConnectedGroup, findKingdoms, findRegions } from '../board'
import { createInitialBoard, Cell, BOARD_ROWS, BOARD_COLS } from '../types'

describe('getNeighbors', () => {
  it('returns 4 neighbors for center cell', () => {
    expect(getNeighbors({ row: 5, col: 5 })).toHaveLength(4)
  })

  it('returns 2 neighbors for corner cell', () => {
    expect(getNeighbors({ row: 0, col: 0 })).toHaveLength(2)
  })

  it('returns 3 neighbors for edge cell', () => {
    expect(getNeighbors({ row: 0, col: 5 })).toHaveLength(3)
  })
})

describe('findConnectedGroup', () => {
  it('finds connected tiles from a starting position', () => {
    const board = createInitialBoard()
    // Starting temple at (row 0, col 10) is isolated — group of 1
    const group = findConnectedGroup(board, { row: 0, col: 10 })
    expect(group).toHaveLength(1)
  })
})

describe('findKingdoms', () => {
  it('returns empty array on initial board (no leaders)', () => {
    const board = createInitialBoard()
    const kingdoms = findKingdoms(board)
    expect(kingdoms).toHaveLength(0)
  })

  it('finds a kingdom when a leader is placed next to a temple', () => {
    const board = createInitialBoard()
    // Place a leader at (row 0, col 9) next to starting temple at (row 0, col 10)
    board[0][9].leader = { color: 'red', dynasty: 'archer' }
    const kingdoms = findKingdoms(board)
    expect(kingdoms).toHaveLength(1)
    expect(kingdoms[0].leaders).toHaveLength(1)
  })
})
```

**Step 2: Run test to verify failure**

**Step 3: Implement board utilities**

Implement `getNeighbors`, `findConnectedGroup` (flood-fill from a position, following tiles and leaders), `findKingdoms` (all connected groups containing at least one leader), `findRegions` (connected groups with no leaders).

Key: flood fill follows adjacency through any cell that has a tile (face-up or face-down), a leader, or both. Catastrophes and empty cells break connectivity.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/engine/board.ts src/engine/__tests__/board.test.ts
git commit -m "add board utilities: adjacency, flood fill, kingdom detection"
```

---

### Task 5: Action Validation

**Files:**
- Create: `src/engine/validation.ts`
- Test: `src/engine/__tests__/validation.test.ts`

**Step 1: Write failing tests**

Test `getValidTilePlacements(state, color)`, `getValidLeaderPlacements(state, color)`, `canPlaceCatastrophe(state, position)`, `canSwapTiles(state)`.

Key validation rules to test:
- Tiles: correct terrain, empty cell, doesn't unite 3+ kingdoms
- Leaders: empty land, adjacent to face-up red temple, doesn't unite 2 kingdoms
- Catastrophe: not on leader, treasure, monument, existing catastrophe

**Step 2: Run test to verify failure**

**Step 3: Implement validation functions**

Each function takes GameState and returns valid positions or boolean. Uses `findKingdoms` to check kingdom-uniting constraints.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/engine/validation.ts src/engine/__tests__/validation.test.ts
git commit -m "add action validation for tiles, leaders, catastrophes"
```

---

### Task 6: Simple Actions (swap tiles, pass, withdraw leader)

**Files:**
- Create: `src/engine/actions.ts`
- Test: `src/engine/__tests__/actions.test.ts`

**Step 1: Write failing tests**

```typescript
describe('swapTiles', () => {
  it('removes selected tiles and draws replacements', () => {
    const state = createGame(2)
    const handBefore = [...state.players[0].hand]
    const result = applyAction(state, { type: 'swapTiles', indices: [0, 1] })
    expect(result.players[0].hand).toHaveLength(6)
    expect(result.actionsRemaining).toBe(1)
  })
})

describe('pass', () => {
  it('decrements actions remaining', () => {
    const state = createGame(2)
    const result = applyAction(state, { type: 'pass' })
    expect(result.actionsRemaining).toBe(1)
  })
})

describe('withdrawLeader', () => {
  it('removes leader from board and returns to player supply', () => {
    // Setup: place a leader on board, then withdraw
    const state = createGame(2)
    state.board[0][9].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders[0].position = { row: 0, col: 9 }
    const result = applyAction(state, { type: 'withdrawLeader', color: 'red' })
    expect(result.board[0][9].leader).toBeNull()
    expect(result.players[0].leaders[0].position).toBeNull()
  })
})
```

**Step 2: Run test to verify failure**

**Step 3: Implement applyAction with these three simple action types**

`applyAction(state: GameState, action: GameAction): GameState` — validates the action, returns new state (immutable). Throws on illegal actions.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/engine/actions.ts src/engine/__tests__/actions.test.ts
git commit -m "add simple actions: swap tiles, pass, withdraw leader"
```

---

### Task 7: Place Tile (no conflict)

**Files:**
- Modify: `src/engine/actions.ts`
- Test: `src/engine/__tests__/placeTile.test.ts`

**Step 1: Write failing tests**

```typescript
describe('placeTile', () => {
  it('places tile on empty land space', () => {
    const state = createTestState()
    // Place black tile on empty land
    const result = applyAction(state, { type: 'placeTile', color: 'black', position: { row: 0, col: 0 } })
    expect(result.board[0][0].tile).toBe('black')
  })

  it('scores 1 VP when matching leader in kingdom', () => {
    // Setup: kingdom with black king, place black tile in it
    // Expect: king's owner gets +1 black VP
  })

  it('king scores 1 black VP when no matching leader', () => {
    // Setup: kingdom with black king only, place green tile in it
    // Expect: king's owner gets +1 black VP
  })

  it('no VP when no leader in kingdom', () => {
    // Place tile into region (no leaders)
    // Expect: no score change
  })

  it('rejects blue tile on land', () => {
    // Expect: throws or returns error
  })

  it('rejects red tile on river', () => {
    // Expect: throws or returns error
  })

  it('removes tile from player hand', () => {
    // Ensure the placed tile is removed from hand
  })
})
```

**Step 2: Run test to verify failure**

**Step 3: Implement placeTile in applyAction**

Handle: validation, board update, hand update, scoring (find kingdom, check for matching leader or king), decrement actions.

Do NOT handle conflict or monument yet — those come in later tasks.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/engine/actions.ts src/engine/__tests__/placeTile.test.ts
git commit -m "add tile placement with scoring (no conflict path yet)"
```

---

### Task 8: Place Leader + Internal Conflict (Revolt)

**Files:**
- Modify: `src/engine/actions.ts`
- Create: `src/engine/conflict.ts`
- Test: `src/engine/__tests__/revolt.test.ts`

**Step 1: Write failing tests**

```typescript
describe('placeLeader', () => {
  it('places leader on empty land adjacent to temple', () => {
    // Valid placement, no conflict
  })

  it('rejects placement not adjacent to face-up red temple', () => {
    // No temple nearby
  })

  it('rejects placement that would unite two kingdoms', () => {
    // Leader position adjacent to tiles from two separate kingdoms
  })
})

describe('revolt', () => {
  it('triggers revolt when placing leader in kingdom with same-colored leader', () => {
    // Setup: kingdom with player 2's red priest
    // Player 1 places red priest in same kingdom
    // State transitions to conflictSupport phase
  })

  it('attacker wins with more adjacent temples + support', () => {
    // Commit support tiles, attacker wins
    // Loser leader removed, winner gets 1 red VP
  })

  it('defender wins ties', () => {
    // Equal strength, defender wins
  })

  it('only counts face-up red temples as base strength', () => {
    // Flipped temples don't count
  })

  it('committed tiles removed from game (not returned to bag)', () => {
    // Both players' committed tiles gone
  })
})
```

**Step 2: Run test to verify failure**

**Step 3: Implement**

- `placeLeader` in actions.ts: validate, check for revolt trigger, either place peacefully or transition to `conflictSupport` phase.
- `conflict.ts`: `resolveRevolt(state, attackerSupport, defenderSupport): GameState` — count adjacent temples, add support, compare, apply aftermath.
- `commitSupport` action handler: collects attacker's tiles first, then defender's, then calls resolve.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/engine/actions.ts src/engine/conflict.ts src/engine/__tests__/revolt.test.ts
git commit -m "add leader placement and internal conflict (revolt) resolution"
```

---

### Task 9: External Conflict (War)

**Files:**
- Modify: `src/engine/actions.ts`, `src/engine/conflict.ts`
- Test: `src/engine/__tests__/war.test.ts`

**Step 1: Write failing tests**

```typescript
describe('war', () => {
  it('triggers war when tile unites two kingdoms with same-colored leaders', () => {
    // Setup: two kingdoms each with a red priest, tile connects them
    // State transitions to warOrderChoice or conflictSupport
  })

  it('active player chooses war order when multiple conflicts', () => {
    // Two kingdoms both have red and green conflicts
    // Active player picks which to resolve first
  })

  it('winner scores 1 per removed tile + 1 for leader', () => {
    // Loser's side has 3 red tiles, winner gets 4 red VP
  })

  it('loser tiles removed from game', () => {
    // All conflict-color tiles on loser side gone
  })

  it('unification tile does not count as supporter', () => {
    // The tile that united kingdoms is not counted
  })

  it('kingdom may split after war', () => {
    // Removing tiles disconnects remaining pieces
  })

  it('leaders without temple adjacency withdrawn after war', () => {
    // Tile removal leaves leader stranded
  })

  it('defender wins ties', () => {
    // Equal strength
  })
})
```

**Step 2: Run test to verify failure**

**Step 3: Implement**

- Detect war trigger in `placeTile`: if tile unites 2 kingdoms, find all same-colored leader pairs, set up pending wars.
- `warOrderChoice` phase: if multiple wars, active player picks order.
- `resolveWar`: count supporters on each side (tiles of conflict color connected to each leader on their side of the unification point), add hand support, compare, remove loser's tiles and leader, score, re-evaluate board.
- After each war resolution, check if remaining wars are still valid (kingdom may have split).

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/engine/actions.ts src/engine/conflict.ts src/engine/__tests__/war.test.ts
git commit -m "add external conflict (war) resolution"
```

---

### Task 10: Monuments

**Files:**
- Create: `src/engine/monument.ts`
- Test: `src/engine/__tests__/monument.test.ts`

**Step 1: Write failing tests**

```typescript
describe('monuments', () => {
  it('detects 2x2 same-color square after tile placement', () => {
    // Place 4th red tile completing a 2x2
    // State transitions to monumentChoice
  })

  it('building flips tiles face-down and places monument', () => {
    // Accept monument build
    // 4 tiles flipped, monument placed
  })

  it('declining monument leaves tiles face-up', () => {
    // Decline, tiles unchanged
  })

  it('monument must have a matching color', () => {
    // If no available monument matches, no option offered
  })

  it('face-down tiles still connect but dont count as supporters', () => {
    // Connectivity preserved, but war strength calculation skips them
  })

  it('leader withdrawn if monument flip removes temple adjacency', () => {
    // Leader next to temples that get flipped, no other temples nearby
  })

  it('monument scoring at end of turn awards VP per matching leader', () => {
    // Player's red priest in kingdom with red-blue monument
    // Gets 1 red VP at end of turn
  })
})
```

**Step 2: Run test to verify failure**

**Step 3: Implement**

- `find2x2Square(board, position): Position | null` — check all 2x2 squares containing the given position for same-color face-up tiles.
- `monumentChoice` phase handler.
- `buildMonument` / `declineMonument` action handlers.
- `scoreMonuments(state): GameState` — called at end of turn.
- Leader withdrawal check after flipping tiles.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/engine/monument.ts src/engine/__tests__/monument.test.ts
git commit -m "add monument building and end-of-turn monument scoring"
```

---

### Task 11: Catastrophe Tiles

**Files:**
- Modify: `src/engine/actions.ts`
- Test: `src/engine/__tests__/catastrophe.test.ts`

**Step 1: Write failing tests**

```typescript
describe('catastrophe', () => {
  it('places catastrophe on empty space', () => {
    // Space becomes permanently blocked
  })

  it('destroys face-up tile when placed on one', () => {
    // Tile removed from game
  })

  it('cannot place on leader, treasure, monument, or existing catastrophe', () => {
    // All should throw/reject
  })

  it('splits kingdom when connectivity broken', () => {
    // Catastrophe in middle of a chain
  })

  it('leader withdrawn if stranded by catastrophe', () => {
    // Leader loses temple adjacency due to split
  })

  it('decrements player catastrophe count', () => {
    // Player had 2, now has 1
  })
})
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "add catastrophe tile placement"
```

---

### Task 12: Turn Flow (draw, treasure, game end)

**Files:**
- Create: `src/engine/turn.ts`
- Test: `src/engine/__tests__/turn.test.ts`

**Step 1: Write failing tests**

```typescript
describe('turn flow', () => {
  it('draws tiles to refill hand to 6 after actions', () => {
    // Player has 4 tiles after 2 placements, draws 2
  })

  it('treasure collected when kingdom has 2+ treasures and trader', () => {
    // Setup: kingdom with trader and 2 treasures
    // Trader owner gets 1 treasure, 1 remains
  })

  it('treasures on starting spaces taken first', () => {
    // Priority rule
  })

  it('no collection without trader', () => {
    // Kingdom with 2 treasures but no green leader
  })

  it('game ends when 1-2 treasures remain on board', () => {
    // State transitions to gameOver
  })

  it('game ends when bag is empty and player needs to draw', () => {
    // Bag exhausted
  })
})

describe('final scoring', () => {
  it('score is minimum across 4 colors', () => {
    // Player: red=5, blue=8, green=3, black=7 → score=3
  })

  it('treasures assigned to weakest color', () => {
    // Player: red=5, blue=8, green=3, black=7, treasures=2 → green=5, score=5
  })

  it('tiebreak by second-lowest, third, fourth', () => {
    // Two players tied on minimum
  })
})
```

**Step 2-5: Implement, test, commit**

- `endTurn(state): GameState` — orchestrates draw, monument scoring, treasure collection, game end check, advance to next player.
- `calculateFinalScores(state): { playerIndex: number; score: number }[]`
- Wire into action flow: when `actionsRemaining` hits 0, auto-call `endTurn`.

```bash
git commit -m "add turn flow: draw, treasure collection, game end, final scoring"
```

---

### Task 13: Game Reducer (unified action dispatcher)

**Files:**
- Create: `src/engine/reducer.ts`
- Test: `src/engine/__tests__/reducer.test.ts`

**Step 1: Write failing tests**

```typescript
describe('gameReducer', () => {
  it('rejects actions from wrong player', () => {
    // Player 1 tries to act on player 0's turn
  })

  it('rejects actions in wrong phase', () => {
    // Try to place tile during conflictSupport phase
  })

  it('full turn: place tile, place leader, draw, next player', () => {
    // Play through a complete turn
  })

  it('full turn with revolt: place leader triggers conflict, resolve, continue', () => {
    // Multi-phase turn
  })
})
```

**Step 2-5: Implement, test, commit**

The reducer is the single entry point that routes actions to the appropriate handler based on `turnPhase` and action type. It validates phase/player, delegates to the specific action handler, and manages phase transitions.

```bash
git commit -m "add unified game reducer with phase management"
```

---

### Task 14: Canvas Board Rendering

**Files:**
- Create: `src/rendering/boardRenderer.ts`
- Create: `src/rendering/colors.ts`
- Create: `src/components/GameBoard.tsx`

**Step 1: Implement color/style constants**

```typescript
// src/rendering/colors.ts
export const COLORS = {
  land: '#e8d5a3',        // sandy
  river: '#4a90d9',       // blue water
  red: '#c0392b',         // temples
  blue: '#2980b9',        // farms
  green: '#27ae60',       // markets
  black: '#2c3e50',       // settlements
  grid: '#bbb',
  highlight: 'rgba(255, 255, 0, 0.3)',
  catastrophe: '#666',
  treasure: '#f1c40f',    // gold
}

export const DYNASTY_COLORS = {
  archer: '#e74c3c',
  bull: '#3498db',
  pot: '#2ecc71',
  lion: '#f39c12',
}
```

**Step 2: Implement boardRenderer**

```typescript
// src/rendering/boardRenderer.ts
// drawBoard(ctx, state, cellSize, highlights?)
// - Draw terrain grid
// - Draw tiles (colored squares)
// - Draw leaders (colored circles with dynasty icon)
// - Draw monuments (2x2 two-tone blocks)
// - Draw catastrophes (X marks)
// - Draw treasures (gold diamonds)
// - Draw highlights for valid placements
```

Pure function: takes canvas 2D context + game state + config, draws everything. No React dependency.

**Step 3: Implement GameBoard React component**

```typescript
// src/components/GameBoard.tsx
// Canvas element with ref
// useEffect to redraw when state changes
// onClick handler: translate pixel to grid cell, call onCellClick prop
// onMouseMove: highlight hovered cell
```

**Step 4: Manual verification**

```bash
npm run dev
# Open browser, verify board renders with terrain and starting temples
```

**Step 5: Commit**

```bash
git add src/rendering/ src/components/GameBoard.tsx
git commit -m "add canvas board rendering with terrain, tiles, leaders, monuments"
```

---

### Task 15: React UI Shell

**Files:**
- Modify: `src/App.tsx`
- Create: `src/components/PlayerPanel.tsx`
- Create: `src/components/HandPanel.tsx`
- Create: `src/components/ActionBar.tsx`
- Create: `src/components/TopBar.tsx`
- Create: `src/hooks/useGame.ts`

**Step 1: Implement useGame hook**

```typescript
// src/hooks/useGame.ts
// Wraps useReducer with gameReducer
// Provides: state, dispatch, selectedTile, selectedLeader, setSelectedTile, setSelectedLeader
// Handles AI turn triggering (useEffect watches for AI player's turn)
```

**Step 2: Implement UI components**

- `TopBar`: current player indicator, turn phase display
- `PlayerPanel`: scores (4 colors + treasures), leader status (on/off board), catastrophes remaining
- `HandPanel`: clickable tiles in hand, selected tile highlighted
- `ActionBar`: context-sensitive buttons (Swap Tiles, Pass, Withdraw Leader, and during conflicts: Commit Support)

**Step 3: Wire up App.tsx**

Layout matching the design doc. GameBoard on the left, panels on the right, bars top and bottom.

**Step 4: Manual verification**

```bash
npm run dev
# Verify layout, panels show initial state, tiles are clickable
```

**Step 5: Commit**

```bash
git add src/App.tsx src/components/ src/hooks/
git commit -m "add React UI shell: player panel, hand, action bar, top bar"
```

---

### Task 16: Interaction Flow (click-to-play)

**Files:**
- Modify: `src/components/GameBoard.tsx`
- Modify: `src/hooks/useGame.ts`
- Create: `src/components/ConflictDialog.tsx`
- Create: `src/components/MonumentDialog.tsx`

**Step 1: Implement click-to-place flow**

- Select tile from hand → click board cell → dispatch `placeTile`
- Click leader from player panel → click board cell → dispatch `placeLeader`
- During `conflictSupport` phase: show ConflictDialog with tile commitment UI
- During `monumentChoice` phase: show MonumentDialog with available monument options
- During `warOrderChoice` phase: show war order picker

**Step 2: Implement ConflictDialog**

Shows attacker vs defender, base strengths, lets current resolver select tiles from hand to commit.

**Step 3: Implement MonumentDialog**

Shows available monuments matching the 2x2 color, build or decline buttons.

**Step 4: Manual verification**

Play through several turns manually in browser.

**Step 5: Commit**

```bash
git add src/components/ src/hooks/
git commit -m "add interactive play flow with conflict and monument dialogs"
```

---

### Task 17: Simple AI

**Files:**
- Create: `src/ai/simpleAI.ts`
- Test: `src/ai/__tests__/simpleAI.test.ts`

**Step 1: Write failing tests**

```typescript
describe('simpleAI', () => {
  it('returns a valid action for the given state', () => {
    const state = createGame(2, [false, true])
    // Advance to AI turn
    const action = getAIAction(state)
    // Should not throw when applied
    expect(() => applyAction(state, action)).not.toThrow()
  })

  it('commits support tiles in conflict when it can win', () => {
    // AI is defender in revolt with enough red tiles to win
  })

  it('does not commit support when it will lose anyway', () => {
    // AI has no red tiles, doesn't waste them
  })
})
```

**Step 2: Run test to verify failure**

**Step 3: Implement simple AI**

```typescript
// src/ai/simpleAI.ts
export function getAIAction(state: GameState): GameAction {
  // Priority:
  // 1. If in conflict phase: decide whether to commit support
  // 2. If in monument choice: always build if possible
  // 3. If in war order choice: pick the color with most advantage
  // 4. Normal action: pick from valid moves:
  //    a. Place tile where own leader scores (prefer weakest color)
  //    b. Place leader if none on board yet
  //    c. Swap tiles if hand is bad (no tiles matching own leaders' colors)
  //    d. Pass as last resort
}
```

Keep it deliberately simple. No lookahead, no minimax.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add src/ai/ 
git commit -m "add simple heuristic AI opponent"
```

---

### Task 18: AI Turn Integration

**Files:**
- Modify: `src/hooks/useGame.ts`
- Modify: `src/App.tsx`

**Step 1: Wire AI into game loop**

In `useGame` hook: when `state.turnPhase === 'action'` and current player `isAI`, trigger AI after a short delay (500ms). Dispatch AI actions sequentially with delays between them so the human can see what happened.

**Step 2: Add visual feedback for AI turns**

- Highlight the cell the AI acted on
- Brief text in TopBar: "AI is thinking..." → "AI placed black tile at (3,5)"

**Step 3: Handle AI in conflict phases**

AI auto-commits support tiles when it's their turn in a conflict.

**Step 4: Manual verification**

Play a full game against AI in browser.

**Step 5: Commit**

```bash
git add src/hooks/useGame.ts src/App.tsx
git commit -m "integrate AI into game loop with visual feedback"
```

---

### Task 19: Game Over Screen

**Files:**
- Create: `src/components/GameOverScreen.tsx`
- Modify: `src/App.tsx`

**Step 1: Implement GameOverScreen**

Shows:
- Each player's scores by color
- Treasure assignments (auto-calculated optimal)
- Final score (minimum color)
- Winner announcement
- "Play Again" button

**Step 2: Implement optimal treasure assignment**

```typescript
// Given scores and N treasures, find assignment that maximizes minimum color
function assignTreasures(score: Record<TileColor, number>, treasures: number): Record<TileColor, number>
```

**Step 3: Wire into App**

When `turnPhase === 'gameOver'`, overlay GameOverScreen.

**Step 4: Manual verification**

**Step 5: Commit**

```bash
git add src/components/GameOverScreen.tsx src/App.tsx
git commit -m "add game over screen with final scoring and treasure assignment"
```

---

### Task 20: Game Setup Screen + Polish

**Files:**
- Create: `src/components/SetupScreen.tsx`
- Modify: `src/App.tsx`
- Modify: `src/index.css`

**Step 1: Implement SetupScreen**

- Choose number of players (2-4)
- For each slot: Human or AI toggle
- Dynasty selection (cosmetic)
- "Start Game" button

**Step 2: Style everything**

Clean up CSS for the full layout. Dark background, clear color coding, readable text.

**Step 3: Wire into App**

App starts with SetupScreen, transitions to game on start.

**Step 4: Full playtest**

Play a complete game start to finish. Verify all rules work correctly.

**Step 5: Commit**

```bash
git add src/components/SetupScreen.tsx src/App.tsx src/index.css
git commit -m "add game setup screen and visual polish"
```

---

## Task Dependency Graph

```
1 (scaffold)
└─ 2 (types)
   └─ 3 (setup)
      └─ 4 (board utils)
         └─ 5 (validation)
            ├─ 6 (simple actions)
            ├─ 7 (place tile)
            ├─ 8 (revolt)
            ├─ 9 (war)
            ├─ 10 (monuments)
            └─ 11 (catastrophe)
               └─ 12 (turn flow)
                  └─ 13 (reducer)
                     ├─ 14 (canvas rendering)
                     │  └─ 16 (interaction flow)
                     ├─ 15 (UI shell)
                     │  └─ 16
                     └─ 17 (AI)
                        └─ 18 (AI integration)
                           └─ 19 (game over)
                              └─ 20 (setup screen + polish)
```
