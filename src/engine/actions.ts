import { GameState, GameAction, TileColor, Position } from './types'
import { findKingdoms } from './board'

export function applyAction(state: GameState, action: GameAction): GameState {
  const next = structuredClone(state)

  // Actions that consume a turn action require actionsRemaining > 0
  if (action.type === 'pass' || action.type === 'swapTiles' || action.type === 'withdrawLeader' || action.type === 'placeTile') {
    if (next.actionsRemaining <= 0) {
      throw new Error('No actions remaining')
    }
  }

  switch (action.type) {
    case 'pass':
      next.actionsRemaining -= 1
      return next

    case 'swapTiles':
      return handleSwapTiles(next, action.indices)

    case 'withdrawLeader':
      return handleWithdrawLeader(next, action.color)

    case 'placeTile':
      return handlePlaceTile(next, action.color, action.position)

    default:
      throw new Error(`Unhandled action type: ${(action as GameAction).type}`)
  }
}

function handleSwapTiles(state: GameState, indices: number[]): GameState {
  const player = state.players[state.currentPlayer]

  // Validate indices
  for (const idx of indices) {
    if (idx < 0 || idx >= player.hand.length) {
      throw new Error(`Invalid tile index: ${idx}`)
    }
  }

  // Remove tiles at indices (process in descending order to preserve indices)
  const sortedIndices = [...indices].sort((a, b) => b - a)
  for (const idx of sortedIndices) {
    // Return tile to bag
    state.bag.push(player.hand[idx])
    player.hand.splice(idx, 1)
  }

  // Draw replacements
  const drawCount = Math.min(indices.length, state.bag.length)
  for (let i = 0; i < drawCount; i++) {
    player.hand.push(state.bag.shift()!)
  }

  state.actionsRemaining -= 1
  return state
}

function handleWithdrawLeader(state: GameState, color: string): GameState {
  const player = state.players[state.currentPlayer]
  const leaderEntry = player.leaders.find(l => l.color === color)

  if (!leaderEntry || !leaderEntry.position) {
    throw new Error(`Leader ${color} is not on the board`)
  }

  const { row, col } = leaderEntry.position
  state.board[row][col].leader = null
  leaderEntry.position = null

  state.actionsRemaining -= 1
  return state
}

function handlePlaceTile(state: GameState, color: TileColor, position: Position): GameState {
  const player = state.players[state.currentPlayer]
  const cell = state.board[position.row][position.col]

  // Validate terrain
  if (color === 'blue' && cell.terrain !== 'river') {
    throw new Error('Blue tiles must be placed on river')
  }
  if (color !== 'blue' && cell.terrain !== 'land') {
    throw new Error('Non-blue tiles must be placed on land')
  }

  // Validate cell is empty
  if (cell.tile !== null || cell.leader !== null || cell.catastrophe || cell.monument !== null) {
    throw new Error('Cell is not empty')
  }

  // Validate player has the tile
  const tileIndex = player.hand.indexOf(color)
  if (tileIndex === -1) {
    throw new Error(`Player does not have a ${color} tile`)
  }

  // Place tile on board
  cell.tile = color

  // Remove tile from hand
  player.hand.splice(tileIndex, 1)

  // Score: find which kingdom the tile is now in
  const kingdoms = findKingdoms(state.board)
  const kingdom = kingdoms.find(k =>
    k.positions.some(p => p.row === position.row && p.col === position.col)
  )

  if (kingdom) {
    // Look for a leader matching the tile color
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
    } else {
      // No matching leader — check for king (black leader)
      const king = kingdom.leaders.find(l => l.color === 'black')
      if (king) {
        const owner = state.players.find(p =>
          p.leaders.some(l => l.color === 'black' && l.position &&
            l.position.row === king.position.row &&
            l.position.col === king.position.col)
        )
        if (owner) {
          owner.score.black += 1
        }
      }
    }
  }

  state.actionsRemaining -= 1
  return state
}
