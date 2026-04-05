import { GameState, GameAction, TileColor, LeaderColor, Position } from '../engine/types'
import { getValidTilePlacements, getValidLeaderPlacements } from '../engine/validation'
import { findKingdoms } from '../engine/board'

const LEADER_COLORS: LeaderColor[] = ['red', 'blue', 'green', 'black']
const TILE_COLORS: TileColor[] = ['red', 'blue', 'green', 'black']

export function getAIAction(state: GameState): GameAction {
  try {
    switch (state.turnPhase) {
      case 'conflictSupport':
        return handleConflictSupport(state)
      case 'monumentChoice':
        return handleMonumentChoice(state)
      case 'warOrderChoice':
        return handleWarOrderChoice(state)
      case 'action':
        return handleActionPhase(state)
      default:
        return { type: 'pass' }
    }
  } catch {
    return { type: 'pass' }
  }
}

function handleConflictSupport(state: GameState): GameAction {
  const conflict = state.pendingConflict
  if (!conflict) return { type: 'commitSupport', indices: [] }

  // Determine who is committing: attacker first, then defender
  const isAttackerTurn = conflict.attackerCommitted === null
  const playerIndex = isAttackerTurn
    ? conflict.attacker.playerIndex
    : conflict.defender.playerIndex
  const player = state.players[playerIndex]
  const supportColor: TileColor = conflict.type === 'revolt' ? 'red' : conflict.color as TileColor

  // Find indices of matching tiles in hand
  const matchingIndices: number[] = []
  for (let i = 0; i < player.hand.length; i++) {
    if (player.hand[i] === supportColor) {
      matchingIndices.push(i)
    }
  }

  // Use already-determined attacker/defender status
  const isAttacker = isAttackerTurn
  const myStrength = isAttacker ? conflict.attackerStrength : conflict.defenderStrength
  const opponentStrength = isAttacker ? conflict.defenderStrength : conflict.attackerStrength

  // If opponent hasn't committed yet, commit all matching tiles to maximize chances
  const opponentCommitted = isAttacker ? conflict.defenderCommitted : conflict.attackerCommitted
  if (opponentCommitted === null) {
    // Don't know opponent's commitment yet - commit all to be safe if we have a chance
    if (matchingIndices.length > 0) {
      return { type: 'commitSupport', indices: matchingIndices }
    }
    return { type: 'commitSupport', indices: [] }
  }

  // Opponent has committed - calculate if we can win
  const opponentTotal = opponentStrength + opponentCommitted.length
  const myPotentialTotal = myStrength + matchingIndices.length

  if (myPotentialTotal > opponentTotal) {
    // We can win - commit just enough to win
    const needed = opponentTotal - myStrength + 1
    const toCommit = matchingIndices.slice(0, Math.max(0, needed))
    return { type: 'commitSupport', indices: toCommit }
  }

  // Can't win - don't waste tiles
  return { type: 'commitSupport', indices: [] }
}

function handleMonumentChoice(state: GameState): GameAction {
  // Always build if possible - monuments are valuable
  const pending = state.pendingMonument
  if (!pending) return { type: 'declineMonument' }

  // Find an available monument that uses the pending color
  for (const monument of state.monuments) {
    if (monument.position !== null) continue // already placed
    if (monument.color1 === pending.color || monument.color2 === pending.color) {
      return { type: 'buildMonument', monumentId: monument.id }
    }
  }

  return { type: 'declineMonument' }
}

function handleWarOrderChoice(state: GameState): GameAction {
  const conflict = state.pendingConflict
  if (!conflict?.pendingWarColors?.length) {
    // Fallback: pick any color
    return { type: 'chooseWarOrder', color: 'red' }
  }

  const player = state.players[state.currentPlayer]
  const pending = conflict.pendingWarColors

  // Pick the color where AI has the most tiles in hand (strongest advantage)
  let bestColor = pending[0]
  let bestCount = -1

  for (const color of pending) {
    const count = player.hand.filter(t => t === color).length
    if (count > bestCount) {
      bestCount = count
      bestColor = color
    }
  }

  return { type: 'chooseWarOrder', color: bestColor }
}

function handleActionPhase(state: GameState): GameAction {
  const player = state.players[state.currentPlayer]
  const onBoardLeaders = player.leaders.filter(l => l.position !== null)

  // Priority 1: Place leaders if none on board
  if (onBoardLeaders.length === 0) {
    const action = tryPlaceLeader(state, player)
    if (action) return action
  }

  // Priority 2: Place tile where own leader scores
  if (onBoardLeaders.length > 0) {
    const action = tryPlaceScoringTile(state, player)
    if (action) return action
  }

  // Priority 3: Place more leaders if opportunity exists
  if (onBoardLeaders.length < 4) {
    const action = tryPlaceLeader(state, player)
    if (action) return action
  }

  // Priority 4: Swap tiles if hand has no tiles matching on-board leaders
  if (onBoardLeaders.length > 0 && player.hand.length > 0) {
    const onBoardColors = new Set(onBoardLeaders.map(l => l.color))
    const hasMatchingTile = player.hand.some(t => onBoardColors.has(t))
    if (!hasMatchingTile) {
      const indices = player.hand.map((_, i) => i)
      return { type: 'swapTiles', indices }
    }
  }

  // Priority 5: Place any valid tile (even if not directly scoring)
  const anyTileAction = tryPlaceAnyTile(state, player)
  if (anyTileAction) return anyTileAction

  // Last resort: pass
  return { type: 'pass' }
}

function tryPlaceLeader(
  state: GameState,
  player: GameState['players'][number],
): GameAction | null {
  // Count tiles per color in hand
  const colorCounts: Record<string, number> = {}
  for (const tile of player.hand) {
    colorCounts[tile] = (colorCounts[tile] || 0) + 1
  }

  // Sort off-board leaders by tile count in hand (prefer colors with more tiles)
  const offBoardLeaders = player.leaders
    .filter(l => l.position === null)
    .sort((a, b) => (colorCounts[b.color] || 0) - (colorCounts[a.color] || 0))

  for (const leader of offBoardLeaders) {
    const validPlacements = getValidLeaderPlacements(state, leader.color)
    if (validPlacements.length > 0) {
      // Pick first valid placement
      return { type: 'placeLeader', color: leader.color, position: validPlacements[0] }
    }
  }

  return null
}

function tryPlaceScoringTile(
  state: GameState,
  player: GameState['players'][number],
): GameAction | null {
  const kingdoms = findKingdoms(state.board)
  const dynasty = player.dynasty
  const onBoardLeaders = player.leaders.filter(l => l.position !== null)

  // Find colors where AI has leaders on board
  const leaderColors = new Set(onBoardLeaders.map(l => l.color))

  // Sort by weakest score first (try to balance VP)
  const sortedColors = [...leaderColors].sort(
    (a, b) => (player.score[a] || 0) - (player.score[b] || 0),
  )

  for (const color of sortedColors) {
    // Check if AI has tiles of this color in hand
    const tileIndex = player.hand.indexOf(color as TileColor)
    if (tileIndex === -1) continue

    const validPlacements = getValidTilePlacements(state, color as TileColor)
    if (validPlacements.length === 0) continue

    // Find a placement adjacent to a kingdom containing AI's leader of this color
    const leaderInfo = onBoardLeaders.find(l => l.color === color)
    if (!leaderInfo?.position) continue

    // Find the kingdom containing this leader
    const leaderKingdom = kingdoms.find(k =>
      k.leaders.some(l => l.dynasty === dynasty && l.color === color),
    )

    if (leaderKingdom) {
      // Prefer placements adjacent to this kingdom
      const kingdomPosSet = new Set(
        leaderKingdom.positions.map(p => `${p.row},${p.col}`),
      )

      const adjacentPlacement = validPlacements.find(pos => {
        const neighbors = getNeighborPositions(pos)
        return neighbors.some(n => kingdomPosSet.has(`${n.row},${n.col}`))
      })

      if (adjacentPlacement) {
        return { type: 'placeTile', color: color as TileColor, position: adjacentPlacement }
      }
    }

    // Fallback: place anywhere valid
    return { type: 'placeTile', color: color as TileColor, position: validPlacements[0] }
  }

  return null
}

function tryPlaceAnyTile(
  state: GameState,
  player: GameState['players'][number],
): GameAction | null {
  for (const color of TILE_COLORS) {
    if (!player.hand.includes(color)) continue
    const placements = getValidTilePlacements(state, color)
    if (placements.length > 0) {
      return { type: 'placeTile', color, position: placements[0] }
    }
  }
  return null
}

function getNeighborPositions(pos: Position): Position[] {
  const { row, col } = pos
  const neighbors: Position[] = []
  if (row > 0) neighbors.push({ row: row - 1, col })
  if (row < 10) neighbors.push({ row: row + 1, col })
  if (col > 0) neighbors.push({ row, col: col - 1 })
  if (col < 15) neighbors.push({ row, col: col + 1 })
  return neighbors
}
