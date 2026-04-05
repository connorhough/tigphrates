import { GameState, GameAction, TileColor, LeaderColor, Position } from './types'
import { findKingdoms, getNeighbors, findConnectedGroup } from './board'
import { countAdjacentTemples, resolveRevolt } from './conflict'

export function applyAction(state: GameState, action: GameAction): GameState {
  const next = structuredClone(state)

  // Actions that consume a turn action require actionsRemaining > 0
  if (action.type === 'pass' || action.type === 'swapTiles' || action.type === 'withdrawLeader' || action.type === 'placeTile' || action.type === 'placeLeader') {
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

    case 'placeLeader':
      return handlePlaceLeader(next, action.color, action.position)

    case 'commitSupport':
      return handleCommitSupport(next, action.indices)

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

function handlePlaceLeader(state: GameState, color: LeaderColor, position: Position): GameState {
  const player = state.players[state.currentPlayer]
  const cell = state.board[position.row][position.col]

  // Must be empty land
  if (cell.terrain !== 'land') {
    throw new Error('Leaders must be placed on land')
  }
  if (cell.tile !== null || cell.leader !== null || cell.catastrophe || cell.monument !== null) {
    throw new Error('Cell is not empty')
  }

  // Leader must not already be on board
  const leaderEntry = player.leaders.find(l => l.color === color)!
  if (leaderEntry.position !== null) {
    throw new Error(`Leader ${color} is already on board`)
  }

  // Must be adjacent to at least one face-up red temple tile
  const neighbors = getNeighbors(position)
  const adjacentToTemple = neighbors.some(n => {
    const c = state.board[n.row][n.col]
    return c.tile === 'red' && !c.tileFlipped
  })
  if (!adjacentToTemple) {
    throw new Error('Leader must be placed adjacent to a face-up red temple tile')
  }

  // Check: would this unite two or more kingdoms?
  const neighborKingdomIds = new Set<string>()
  const kingdoms = findKingdoms(state.board)
  for (const neighbor of neighbors) {
    for (const kingdom of kingdoms) {
      if (kingdom.leaders.length > 0 && kingdom.positions.some(p => p.row === neighbor.row && p.col === neighbor.col)) {
        const id = kingdom.positions.map(p => `${p.row},${p.col}`).sort().join('|')
        neighborKingdomIds.add(id)
      }
    }
  }
  if (neighborKingdomIds.size > 1) {
    throw new Error('Placing leader would unite two or more kingdoms')
  }

  // Place the leader
  cell.leader = { color, dynasty: player.dynasty }
  leaderEntry.position = position

  // Check for revolt: is this leader now in a kingdom with another leader of the same color?
  const updatedKingdoms = findKingdoms(state.board)
  const kingdom = updatedKingdoms.find(k =>
    k.positions.some(p => p.row === position.row && p.col === position.col)
  )

  if (kingdom) {
    const sameColorLeaders = kingdom.leaders.filter(l => l.color === color)
    if (sameColorLeaders.length === 2) {
      // Revolt! Attacker = player who just placed, defender = existing leader's owner
      const defenderLeader = sameColorLeaders.find(l =>
        l.position.row !== position.row || l.position.col !== position.col
      )!
      const defenderPlayerIndex = state.players.findIndex(p =>
        p.leaders.some(l => l.color === color && l.position &&
          l.position.row === defenderLeader.position.row &&
          l.position.col === defenderLeader.position.col)
      )

      const attackerStrength = countAdjacentTemples(state.board, position)
      const defenderStrength = countAdjacentTemples(state.board, defenderLeader.position)

      state.pendingConflict = {
        type: 'revolt',
        color,
        attacker: { playerIndex: state.currentPlayer, position },
        defender: { playerIndex: defenderPlayerIndex, position: defenderLeader.position },
        attackerStrength,
        defenderStrength,
        attackerCommitted: null,
        defenderCommitted: null,
      }
      state.turnPhase = 'conflictSupport'
      return state
    }
  }

  state.actionsRemaining -= 1
  return state
}

function handleCommitSupport(state: GameState, indices: number[]): GameState {
  if (state.turnPhase !== 'conflictSupport' || !state.pendingConflict) {
    throw new Error('Not in conflict support phase')
  }

  const conflict = state.pendingConflict

  // Determine who is committing
  const isAttackerTurn = conflict.attackerCommitted === null
  const committingPlayerIndex = isAttackerTurn
    ? conflict.attacker.playerIndex
    : conflict.defender.playerIndex
  const player = state.players[committingPlayerIndex]

  // Validate indices and tile colors
  for (const idx of indices) {
    if (idx < 0 || idx >= player.hand.length) {
      throw new Error(`Invalid tile index: ${idx}`)
    }
    if (conflict.type === 'revolt' && player.hand[idx] !== 'red') {
      throw new Error('Only red tiles can be committed during a revolt')
    }
  }

  if (isAttackerTurn) {
    conflict.attackerCommitted = indices
    return state
  } else {
    conflict.defenderCommitted = indices
    return resolveRevolt(state)
  }
}
