import { GameState, Position, TileColor, Monument, MonumentId, Cell, BOARD_ROWS, BOARD_COLS } from './types'
import { getNeighbors, findKingdoms } from './board'
import { withdrawStrandedLeaders } from './conflict'

/**
 * Check if placing a tile created a 2x2 same-color square.
 * Returns the top-left position and color, or null.
 */
export function find2x2Square(board: Cell[][], position: Position): { topLeft: Position; color: TileColor } | null {
  const { row, col } = position

  // Check all 4 possible 2x2 squares that include (row, col)
  // Each identified by top-left corner
  const candidates: Position[] = [
    { row: row - 1, col: col - 1 },
    { row: row - 1, col },
    { row, col: col - 1 },
    { row, col },
  ]

  for (const topLeft of candidates) {
    if (topLeft.row < 0 || topLeft.col < 0 ||
        topLeft.row + 1 >= BOARD_ROWS || topLeft.col + 1 >= BOARD_COLS) {
      continue
    }

    const cells = [
      board[topLeft.row][topLeft.col],
      board[topLeft.row][topLeft.col + 1],
      board[topLeft.row + 1][topLeft.col],
      board[topLeft.row + 1][topLeft.col + 1],
    ]

    // All must have face-up tiles of the same color, no monument already there
    if (cells.every(c => c.tile !== null && !c.tileFlipped && c.monument === null)) {
      const color = cells[0].tile!
      if (cells.every(c => c.tile === color)) {
        return { topLeft, color }
      }
    }
  }

  return null
}

/**
 * Get available (unplaced) monuments that match the given color.
 */
export function getAvailableMonuments(monuments: Monument[], color: TileColor): Monument[] {
  return monuments.filter(m =>
    m.position === null && (m.color1 === color || m.color2 === color)
  )
}

/**
 * Build a monument: flip tiles face-down, place monument on cells,
 * update monument position, and withdraw stranded leaders.
 */
export function buildMonument(state: GameState, monumentId: MonumentId, topLeft: Position): GameState {
  const monument = state.monuments.find(m => m.id === monumentId)
  if (!monument) {
    throw new Error(`Monument ${monumentId} not found`)
  }
  if (monument.position !== null) {
    throw new Error(`Monument ${monumentId} is already placed`)
  }

  const positions = [
    { row: topLeft.row, col: topLeft.col },
    { row: topLeft.row, col: topLeft.col + 1 },
    { row: topLeft.row + 1, col: topLeft.col },
    { row: topLeft.row + 1, col: topLeft.col + 1 },
  ]

  // Flip tiles face-down and place monument
  for (const pos of positions) {
    const cell = state.board[pos.row][pos.col]
    cell.tileFlipped = true
    cell.monument = monumentId
  }

  // Update monument position
  monument.position = topLeft

  // Withdraw leaders that lost temple adjacency due to flipping
  withdrawStrandedLeaders(state)

  return state
}

/**
 * Score monuments at end of turn.
 * For each placed monument, if a player has a leader of color1/color2
 * in the same kingdom, they get +1 VP of that color.
 */
export function scoreMonuments(state: GameState): GameState {
  const kingdoms = findKingdoms(state.board)

  for (const monument of state.monuments) {
    if (!monument.position) continue

    // Find which kingdom this monument is in
    const monumentPos = monument.position
    const kingdom = kingdoms.find(k =>
      k.positions.some(p => p.row === monumentPos.row && p.col === monumentPos.col)
    )
    if (!kingdom) continue

    // For each color on the monument, find a matching leader in this kingdom
    for (const color of [monument.color1, monument.color2]) {
      const matchingLeader = kingdom.leaders.find(l => l.color === color)
      if (matchingLeader) {
        // Find the player who owns this leader
        const owner = state.players.find(p =>
          p.leaders.some(l => l.color === matchingLeader.color && l.position &&
            l.position.row === matchingLeader.position.row &&
            l.position.col === matchingLeader.position.col)
        )
        if (owner) {
          owner.score[color] += 1
        }
      }
    }
  }

  return state
}
