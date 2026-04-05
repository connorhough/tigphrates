import { GameState, GameAction } from './types'

export function applyAction(state: GameState, action: GameAction): GameState {
  const next = structuredClone(state)

  // Actions that consume a turn action require actionsRemaining > 0
  if (action.type === 'pass' || action.type === 'swapTiles' || action.type === 'withdrawLeader') {
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
