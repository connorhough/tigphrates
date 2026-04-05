# Tigris & Euphrates Web Game - Tech Stack & Architecture Design

## Tech Stack
- **React + TypeScript** (UI framework)
- **Canvas 2D** (board rendering)
- **Vite** (build tool)
- **React Context + useReducer** (state management)
- **No backend** -- all logic runs client-side

## Architecture

```
src/
  engine/        -- Pure TS game logic (no React dependencies)
  ai/            -- AI opponent logic
  rendering/     -- Canvas drawing code
  components/    -- React components (UI chrome, panels, menus)
  App.tsx        -- Root component
```

### Key Separation
`engine/` is a pure state machine: takes game state + action, returns new game state. Zero UI dependencies, testable in isolation, portable to a server later for online multiplayer.

`rendering/` reads game state and draws the board on canvas. React owns surrounding UI (player hand, score panel, action buttons, conflict dialogs) as DOM components beside/overlaying the canvas.

### Data Flow
React holds game state in `useReducer`. User interactions dispatch actions to the reducer, which delegates to the engine. Engine validates, resolves effects, returns new state. React re-renders, canvas redraws.

AI turns: async function evaluates board, returns actions dispatched through the same reducer.

## Game State Model

```typescript
interface GameState {
  board: Cell[][]              // 16x11 grid
  players: Player[]            // 2-4 players
  bag: TileColor[]             // remaining tiles to draw
  monuments: Monument[]        // placed monuments
  availableMonuments: Monument[] // unplaced monuments
  treasures: TreasurePosition[] // treasures on the board
  currentPlayer: number        // index into players
  actionsRemaining: number     // 0-2
  turnPhase: TurnPhase         // 'action' | 'conflict' | 'monumentChoice' | 'draw' | 'scoring' | 'gameOver'
  pendingConflict: ConflictState | null
}

interface Cell {
  terrain: 'land' | 'river'
  tile: TileColor | null       // red/blue/green/black, null if empty
  tileFlipped: boolean         // face-down under monument
  leader: Leader | null
  catastrophe: boolean
  monument: MonumentId | null
  hasTreasure: boolean
}

interface Player {
  dynasty: Dynasty             // archer/bull/pot/lion
  hand: TileColor[]            // up to 6 tiles
  leaders: LeaderPosition[]    // 4 leaders, each on/off board
  catastrophesRemaining: number
  score: { red: number; blue: number; green: number; black: number }
  treasures: number
  isAI: boolean
}
```

`TurnPhase` models the fact that the game pauses mid-action for conflict resolution or monument decisions. The reducer only accepts actions valid for the current phase.

## Engine Actions

```typescript
type GameAction =
  | { type: 'placeLeader'; color: LeaderColor; position: Position }
  | { type: 'withdrawLeader'; color: LeaderColor }
  | { type: 'placeTile'; color: TileColor; position: Position }
  | { type: 'placeCatastrophe'; position: Position }
  | { type: 'swapTiles'; tiles: number[] }
  | { type: 'commitSupport'; tiles: number[] }
  | { type: 'chooseWarOrder'; color: LeaderColor }
  | { type: 'buildMonument'; monumentId: MonumentId; position: Position }
  | { type: 'declineMonument' }
  | { type: 'pass' }
```

Reducer: `(state: GameState, action: GameAction) => GameState`

Validates legality, applies effects, advances turn phase.

## UI Layout

```
+------------------------------------------+
|  Top Bar: current player, turn phase     |
+------------------+-----------------------+
|                  |  Player Panel          |
|                  |  - Score (4 colors)    |
|   Canvas         |  - Treasures           |
|   (game board)   |  - Leaders (on/off)    |
|                  |  - Catastrophes left   |
|                  +-----------------------+
|                  |  Hand                  |
|                  |  (tile selection)      |
+------------------+-----------------------+
|  Action Bar: context-sensitive buttons   |
+------------------------------------------+
```

Canvas: grid, terrain, tiles, leaders, monuments, catastrophes, treasures, valid placement highlights.

React DOM: player hand, scores, action buttons, conflict UI, monument choice, game-over screen.

Click on canvas translates pixel coords to grid cell, dispatches action based on current phase and selection.

## AI Design

Simple heuristic-based AI to start:

1. Pick valid moves that score points (place tiles where own leaders exist)
2. Prefer tiles matching weakest score color
3. Basic conflict response (commit support if likely to win)
4. Short artificial delay so turns feel natural

No lookahead, no complex strategy. Improve later.

## Future Considerations
- Online multiplayer: move engine to server, add WebSocket layer
- AI difficulty levels
- Advanced board variant
- Animations and sound
