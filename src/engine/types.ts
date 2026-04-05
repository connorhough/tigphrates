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
