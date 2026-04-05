import { GameState, Position, LeaderColor, TileColor } from './types'
import { getNeighbors, findKingdoms } from './board'

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

/**
 * Count tiles of a given color reachable from a leader's position,
 * flood-filling through the merged kingdom but NOT crossing through
 * the unification tile position or the opposing leader's position.
 */
export function countWarSupportTiles(
  board: GameState['board'],
  leaderPos: Position,
  warColor: TileColor,
  unificationTilePos: Position,
  opposingLeaderPos: Position,
): number {
  const visited = new Set<string>()
  const queue: Position[] = []
  let count = 0

  // Start from the leader position — don't count the leader cell itself
  const startKey = `${leaderPos.row},${leaderPos.col}`
  visited.add(startKey)

  // Also block the unification tile and opposing leader
  visited.add(`${unificationTilePos.row},${unificationTilePos.col}`)
  visited.add(`${opposingLeaderPos.row},${opposingLeaderPos.col}`)

  // Seed with neighbors of the leader
  for (const neighbor of getNeighbors(leaderPos)) {
    const key = `${neighbor.row},${neighbor.col}`
    if (visited.has(key)) continue
    visited.add(key)
    const cell = board[neighbor.row][neighbor.col]
    if (cell.tile !== null || cell.leader !== null) {
      queue.push(neighbor)
      if (cell.tile === warColor && !cell.tileFlipped) count++
    }
  }

  // BFS flood fill
  while (queue.length > 0) {
    const pos = queue.shift()!
    for (const neighbor of getNeighbors(pos)) {
      const key = `${neighbor.row},${neighbor.col}`
      if (visited.has(key)) continue
      visited.add(key)
      const cell = board[neighbor.row][neighbor.col]
      if (cell.tile !== null || cell.leader !== null) {
        queue.push(neighbor)
        if (cell.tile === warColor && !cell.tileFlipped) count++
      }
    }
  }

  return count
}

/**
 * Resolve a war after both sides have committed support.
 * Mutates the passed-in state (caller should have already cloned).
 */
export function resolveWar(state: GameState): GameState {
  const conflict = state.pendingConflict!
  const { attacker, defender, attackerCommitted, defenderCommitted } = conflict
  const warColor = conflict.color as TileColor

  const attackerTotal = conflict.attackerStrength + (attackerCommitted?.length ?? 0)
  const defenderTotal = conflict.defenderStrength + (defenderCommitted?.length ?? 0)

  // Defender wins ties
  const attackerWins = attackerTotal > defenderTotal

  const winner = attackerWins ? attacker : defender
  const loser = attackerWins ? defender : attacker

  // Remove loser's leader from board
  state.board[loser.position.row][loser.position.col].leader = null
  const loserPlayer = state.players[loser.playerIndex]
  const loserLeaderEntry = loserPlayer.leaders.find(l => l.color === conflict.color)!
  loserLeaderEntry.position = null

  // Remove all tiles of the war color on the loser's side
  // Flood fill from loser's (now empty) leader position, but since the leader is removed
  // we need to find tiles that were on the loser's side. Use the same logic as counting
  // but collect positions instead.
  const unificationPos = conflict.unificationTilePosition!
  const loserSideTiles = collectWarColorTiles(
    state.board,
    loser.position,
    warColor,
    unificationPos,
    winner.position,
  )

  // Remove those tiles from the board
  for (const pos of loserSideTiles) {
    state.board[pos.row][pos.col].tile = null
    state.board[pos.row][pos.col].hasTreasure = false
  }

  // Winner scores: removed tiles + 1 (for the leader)
  state.players[winner.playerIndex].score[warColor] += loserSideTiles.length + 1

  // Remove committed tiles from both players' hands
  removeCommittedTiles(state, attacker.playerIndex, attackerCommitted ?? [])
  removeCommittedTiles(state, defender.playerIndex, defenderCommitted ?? [])

  // Check for stranded leaders: any leader not adjacent to a face-up red temple
  withdrawStrandedLeaders(state)

  // Check if there are pending wars remaining
  const pendingWarColors = conflict.pendingWarColors ?? []
  // Filter out wars that are no longer valid (both leaders must still be on board)
  const validPendingWars = pendingWarColors.filter(color => {
    const kingdoms = findKingdoms(state.board)
    // Find leaders of this color from different players
    const leadersOnBoard: { playerIndex: number; position: Position }[] = []
    for (let pi = 0; pi < state.players.length; pi++) {
      const entry = state.players[pi].leaders.find(l => l.color === color)
      if (entry && entry.position) {
        leadersOnBoard.push({ playerIndex: pi, position: entry.position })
      }
    }
    if (leadersOnBoard.length < 2) return false
    // Check they're in the same kingdom
    const kingdom = kingdoms.find(k =>
      k.positions.some(p => p.row === leadersOnBoard[0].position.row && p.col === leadersOnBoard[0].position.col)
    )
    if (!kingdom) return false
    return kingdom.positions.some(p =>
      p.row === leadersOnBoard[1].position.row && p.col === leadersOnBoard[1].position.col
    )
  })

  if (validPendingWars.length === 0) {
    state.pendingConflict = null
    state.turnPhase = 'action'
    state.actionsRemaining -= 1
  } else if (validPendingWars.length === 1) {
    // Set up the next war directly
    setupWarConflict(state, validPendingWars[0], unificationPos, [])
    state.turnPhase = 'conflictSupport'
  } else {
    // Multiple wars remain — player chooses
    state.pendingConflict = {
      type: 'war',
      color: validPendingWars[0], // placeholder, will be set by chooseWarOrder
      attacker: conflict.attacker,
      defender: conflict.defender,
      attackerStrength: 0,
      defenderStrength: 0,
      attackerCommitted: null,
      defenderCommitted: null,
      pendingWarColors: validPendingWars,
      unificationTilePosition: unificationPos,
    }
    state.turnPhase = 'warOrderChoice'
  }

  return state
}

/**
 * Collect positions of tiles of the given color on the loser's side.
 * The loser's leader has been removed but we flood fill from their position.
 */
function collectWarColorTiles(
  board: GameState['board'],
  loserLeaderPos: Position,
  warColor: TileColor,
  unificationTilePos: Position,
  winnerLeaderPos: Position,
): Position[] {
  const visited = new Set<string>()
  const queue: Position[] = []
  const result: Position[] = []

  visited.add(`${loserLeaderPos.row},${loserLeaderPos.col}`)
  visited.add(`${unificationTilePos.row},${unificationTilePos.col}`)
  visited.add(`${winnerLeaderPos.row},${winnerLeaderPos.col}`)

  // Seed with neighbors of the loser leader position
  for (const neighbor of getNeighbors(loserLeaderPos)) {
    const key = `${neighbor.row},${neighbor.col}`
    if (visited.has(key)) continue
    visited.add(key)
    const cell = board[neighbor.row][neighbor.col]
    if (cell.tile !== null || cell.leader !== null) {
      queue.push(neighbor)
      if (cell.tile === warColor && !cell.tileFlipped) result.push(neighbor)
    }
  }

  while (queue.length > 0) {
    const pos = queue.shift()!
    for (const neighbor of getNeighbors(pos)) {
      const key = `${neighbor.row},${neighbor.col}`
      if (visited.has(key)) continue
      visited.add(key)
      const cell = board[neighbor.row][neighbor.col]
      if (cell.tile !== null || cell.leader !== null) {
        queue.push(neighbor)
        if (cell.tile === warColor && !cell.tileFlipped) result.push(neighbor)
      }
    }
  }

  return result
}

/**
 * Set up the pendingConflict for a specific war color.
 */
export function setupWarConflict(
  state: GameState,
  warColor: LeaderColor,
  unificationTilePos: Position,
  pendingWarColors: LeaderColor[],
): void {
  // Find the two leaders of this color
  const leadersOnBoard: { playerIndex: number; position: Position }[] = []
  for (let pi = 0; pi < state.players.length; pi++) {
    const entry = state.players[pi].leaders.find(l => l.color === warColor)
    if (entry && entry.position) {
      leadersOnBoard.push({ playerIndex: pi, position: entry.position })
    }
  }

  // Active player's leader is the attacker if they have one; otherwise next player clockwise
  let attackerEntry = leadersOnBoard.find(l => l.playerIndex === state.currentPlayer)
  if (!attackerEntry) {
    // Find next player clockwise from current player
    for (let offset = 1; offset < state.players.length; offset++) {
      const pi = (state.currentPlayer + offset) % state.players.length
      attackerEntry = leadersOnBoard.find(l => l.playerIndex === pi)
      if (attackerEntry) break
    }
  }
  if (!attackerEntry) {
    throw new Error('No attacker found for war')
  }
  const defenderEntry = leadersOnBoard.find(l => l !== attackerEntry)!

  const attackerStrength = countWarSupportTiles(
    state.board,
    attackerEntry.position,
    warColor as TileColor,
    unificationTilePos,
    defenderEntry.position,
  )
  const defenderStrength = countWarSupportTiles(
    state.board,
    defenderEntry.position,
    warColor as TileColor,
    unificationTilePos,
    attackerEntry.position,
  )

  state.pendingConflict = {
    type: 'war',
    color: warColor,
    attacker: { playerIndex: attackerEntry.playerIndex, position: attackerEntry.position },
    defender: { playerIndex: defenderEntry.playerIndex, position: defenderEntry.position },
    attackerStrength,
    defenderStrength,
    attackerCommitted: null,
    defenderCommitted: null,
    pendingWarColors,
    unificationTilePosition: unificationTilePos,
  }
}

/**
 * Withdraw any leaders that are not adjacent to a face-up red temple tile.
 */
export function withdrawStrandedLeaders(state: GameState): void {
  for (const player of state.players) {
    for (const leaderEntry of player.leaders) {
      if (!leaderEntry.position) continue
      const pos = leaderEntry.position
      const neighbors = getNeighbors(pos)
      const adjacentToTemple = neighbors.some(n => {
        const cell = state.board[n.row][n.col]
        return cell.tile === 'red' && !cell.tileFlipped
      })
      if (!adjacentToTemple) {
        state.board[pos.row][pos.col].leader = null
        leaderEntry.position = null
      }
    }
  }
}
