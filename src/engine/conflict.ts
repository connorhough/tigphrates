import { GameState, Position } from './types'
import { getNeighbors } from './board'

/**
 * Count face-up red temple tiles adjacent to the given position.
 */
export function countAdjacentTemples(board: GameState['board'], position: Position): number {
  let count = 0
  for (const neighbor of getNeighbors(position)) {
    const cell = board[neighbor.row][neighbor.col]
    if (cell.tile === 'red' && !cell.tileFlipped) {
      count++
    }
  }
  return count
}

/**
 * Resolve a revolt after both sides have committed support.
 * Mutates the passed-in state (caller should have already cloned).
 */
export function resolveRevolt(state: GameState): GameState {
  const conflict = state.pendingConflict!
  const { attacker, defender, attackerCommitted, defenderCommitted } = conflict

  const attackerTotal = conflict.attackerStrength + (attackerCommitted?.length ?? 0)
  const defenderTotal = conflict.defenderStrength + (defenderCommitted?.length ?? 0)

  // Defender wins ties
  const attackerWins = attackerTotal > defenderTotal

  const winner = attackerWins ? attacker : defender
  const loser = attackerWins ? defender : attacker
  const winnerPlayerIndex = winner.playerIndex

  // Remove loser's leader from board
  state.board[loser.position.row][loser.position.col].leader = null
  const loserPlayer = state.players[loser.playerIndex]
  const loserLeaderEntry = loserPlayer.leaders.find(l => l.color === conflict.color)!
  loserLeaderEntry.position = null

  // Winner gets 1 red VP
  state.players[winnerPlayerIndex].score.red += 1

  // Remove committed tiles from both players' hands (removed from game, not returned to bag)
  removeCommittedTiles(state, attacker.playerIndex, attackerCommitted ?? [])
  removeCommittedTiles(state, defender.playerIndex, defenderCommitted ?? [])

  // Clean up conflict state
  state.pendingConflict = null
  state.turnPhase = 'action'
  state.actionsRemaining -= 1

  return state
}

function removeCommittedTiles(state: GameState, playerIndex: number, indices: number[]): void {
  if (indices.length === 0) return
  const player = state.players[playerIndex]
  // Remove in descending index order to preserve indices
  const sorted = [...indices].sort((a, b) => b - a)
  for (const idx of sorted) {
    player.hand.splice(idx, 1)
  }
}
