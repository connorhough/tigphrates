import { GameState, Position, TileColor, BOARD_ROWS, BOARD_COLS } from './types'
import { getNeighbors, findKingdoms } from './board'

export function getValidTilePlacements(state: GameState, color: TileColor): Position[] {
  const validPositions: Position[] = []
  const requiredTerrain = color === 'blue' ? 'river' : 'land'

  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const cell = state.board[row][col]

      if (cell.terrain !== requiredTerrain) continue
      if (cell.tile !== null) continue
      if (cell.leader !== null) continue
      if (cell.catastrophe) continue
      if (cell.monument !== null) continue

      // Check if placing here would unite 3+ kingdoms
      if (wouldUniteThreeOrMoreKingdoms(state, { row, col })) continue

      validPositions.push({ row, col })
    }
  }

  return validPositions
}

function wouldUniteThreeOrMoreKingdoms(state: GameState, pos: Position): boolean {
  // Find which existing kingdoms are adjacent to this position
  const kingdoms = findKingdoms(state.board)
  if (kingdoms.length < 3) return false

  const adjacentKingdomIndices = new Set<number>()
  const neighbors = getNeighbors(pos)

  for (const neighbor of neighbors) {
    for (let i = 0; i < kingdoms.length; i++) {
      const kingdom = kingdoms[i]
      if (kingdom.positions.some(p => p.row === neighbor.row && p.col === neighbor.col)) {
        adjacentKingdomIndices.add(i)
      }
    }
  }

  return adjacentKingdomIndices.size >= 3
}

export function getValidLeaderPlacements(state: GameState, _color: string): Position[] {
  const validPositions: Position[] = []

  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const cell = state.board[row][col]

      // Must be empty land cell
      if (cell.terrain !== 'land') continue
      if (cell.tile !== null) continue
      if (cell.leader !== null) continue
      if (cell.catastrophe) continue
      if (cell.monument !== null) continue

      // Must be adjacent to at least one face-up red temple tile
      const neighbors = getNeighbors({ row, col })
      const adjacentToTemple = neighbors.some(n => {
        const c = state.board[n.row][n.col]
        return c.tile === 'red' && !c.tileFlipped
      })
      if (!adjacentToTemple) continue

      // Must not unite 2+ kingdoms
      if (wouldLeaderUniteTwoOrMoreKingdoms(state, { row, col })) continue

      validPositions.push({ row, col })
    }
  }

  return validPositions
}

function wouldLeaderUniteTwoOrMoreKingdoms(state: GameState, pos: Position): boolean {
  // A leader connects to adjacent tiles/leaders. If adjacent cells belong to
  // two different kingdoms, placing a leader here would unite them.
  const kingdoms = findKingdoms(state.board)
  if (kingdoms.length < 2) return false

  const adjacentKingdomIndices = new Set<number>()
  const neighbors = getNeighbors(pos)

  for (const neighbor of neighbors) {
    for (let i = 0; i < kingdoms.length; i++) {
      const kingdom = kingdoms[i]
      if (kingdom.positions.some(p => p.row === neighbor.row && p.col === neighbor.col)) {
        adjacentKingdomIndices.add(i)
      }
    }
  }

  return adjacentKingdomIndices.size >= 2
}

export function canPlaceCatastrophe(state: GameState, position: Position): boolean {
  const cell = state.board[position.row][position.col]

  if (cell.leader !== null) return false
  if (cell.hasTreasure) return false
  if (cell.monument !== null) return false
  if (cell.catastrophe) return false

  return true
}

export function canSwapTiles(state: GameState): boolean {
  return state.players[state.currentPlayer].hand.length > 0
}
