import { GameState, GameAction, TileColor, LeaderColor, Position } from '../engine/types'
import { getValidTilePlacements, getValidLeaderPlacements, canPlaceCatastrophe } from '../engine/validation'
import { findKingdoms, getNeighbors } from '../engine/board'
import { countAdjacentTemples, countWarSupportTiles } from '../engine/conflict'

const CATASTROPHE_MIN_TARGET_TILES = 3

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

  const isAttackerTurn = conflict.attackerCommitted === null
  const playerIndex = isAttackerTurn
    ? conflict.attacker.playerIndex
    : conflict.defender.playerIndex
  const player = state.players[playerIndex]
  const supportColor: TileColor = conflict.type === 'revolt' ? 'red' : conflict.color as TileColor

  const matchingIndices: number[] = []
  for (let i = 0; i < player.hand.length; i++) {
    if (player.hand[i] === supportColor) {
      matchingIndices.push(i)
    }
  }

  const isAttacker = isAttackerTurn
  const myStrength = isAttacker ? conflict.attackerStrength : conflict.defenderStrength
  const opponentStrength = isAttacker ? conflict.defenderStrength : conflict.attackerStrength
  const opponentCommitted = isAttacker ? conflict.defenderCommitted : conflict.attackerCommitted

  if (opponentCommitted === null) {
    // Attacker committing first. Bound defender's worst-case commit by their
    // visible matching-color hand count. Commit minimum to beat that ceiling;
    // otherwise commit zero so we don't burn hand on an unwinnable conflict.
    const oppPlayerIndex = isAttacker
      ? conflict.defender.playerIndex
      : conflict.attacker.playerIndex
    const oppMaxCommit = state.players[oppPlayerIndex].hand
      .filter(t => t === supportColor).length
    const oppCeiling = opponentStrength + oppMaxCommit
    const myMaxTotal = myStrength + matchingIndices.length

    if (myMaxTotal > oppCeiling) {
      const needed = oppCeiling - myStrength + 1
      const toCommit = matchingIndices.slice(0, Math.max(0, needed))
      return { type: 'commitSupport', indices: toCommit }
    }
    // Cannot beat defender's worst case. Defender wins ties — don't waste tiles.
    return { type: 'commitSupport', indices: [] }
  }

  // Opponent already committed — exact calculation.
  const opponentTotal = opponentStrength + opponentCommitted.length
  const myPotentialTotal = myStrength + matchingIndices.length

  if (myPotentialTotal > opponentTotal) {
    const needed = opponentTotal - myStrength + 1
    const toCommit = matchingIndices.slice(0, Math.max(0, needed))
    return { type: 'commitSupport', indices: toCommit }
  }

  return { type: 'commitSupport', indices: [] }
}

function handleMonumentChoice(state: GameState): GameAction {
  const pending = state.pendingMonument
  if (!pending) return { type: 'declineMonument' }

  for (const monument of state.monuments) {
    if (monument.position !== null) continue
    if (monument.color1 === pending.color || monument.color2 === pending.color) {
      return { type: 'buildMonument', monumentId: monument.id }
    }
  }

  return { type: 'declineMonument' }
}

function handleWarOrderChoice(state: GameState): GameAction {
  const conflict = state.pendingConflict
  if (!conflict?.pendingWarColors?.length) {
    return { type: 'chooseWarOrder', color: 'red' }
  }

  const player = state.players[state.currentPlayer]
  const pending = conflict.pendingWarColors
  const unificationPos = conflict.unificationTilePosition

  let bestColor: LeaderColor = pending[0]
  let bestScore = -Infinity

  for (const color of pending) {
    // Find the two opposing leaders of this color.
    const leadersOnBoard: { playerIndex: number; pos: Position }[] = []
    for (let pi = 0; pi < state.players.length; pi++) {
      const entry = state.players[pi].leaders.find(l => l.color === color)
      if (entry?.position) {
        leadersOnBoard.push({ playerIndex: pi, pos: entry.position })
      }
    }

    let myAdvantage: number
    if (unificationPos && leadersOnBoard.length >= 2) {
      const attackerEntry =
        leadersOnBoard.find(l => l.playerIndex === state.currentPlayer) ?? leadersOnBoard[0]
      const defenderEntry = leadersOnBoard.find(l => l !== attackerEntry)!
      const attackerStrength = countWarSupportTiles(
        state.board,
        attackerEntry.pos,
        color as TileColor,
        unificationPos,
        defenderEntry.pos,
      )
      const defenderStrength = countWarSupportTiles(
        state.board,
        defenderEntry.pos,
        color as TileColor,
        unificationPos,
        attackerEntry.pos,
      )
      const myHandCount = player.hand.filter(t => t === color).length
      myAdvantage = (attackerStrength + myHandCount) - defenderStrength
    } else {
      // Fallback: hand count only.
      myAdvantage = player.hand.filter(t => t === color).length
    }

    if (myAdvantage > bestScore) {
      bestScore = myAdvantage
      bestColor = color
    }
  }

  return { type: 'chooseWarOrder', color: bestColor }
}

function handleActionPhase(state: GameState): GameAction {
  const player = state.players[state.currentPlayer]
  const onBoardLeaders = player.leaders.filter(l => l.position !== null)

  if (onBoardLeaders.length === 0) {
    const action = tryPlaceLeader(state, player)
    if (action) return action
  }

  if (onBoardLeaders.length > 0) {
    const action = tryPlaceScoringTile(state, player)
    if (action) return action
  }

  if (onBoardLeaders.length < 4) {
    const action = tryPlaceLeader(state, player)
    if (action) return action
  }

  // Catastrophe: only when there's a high-value disruption target. Picks
  // an opponent kingdom where they have a leader collecting VP for a color
  // we cannot contest (we have no leader of that color), and removes the
  // highest-leverage face-up tile.
  if (player.catastrophesRemaining > 0) {
    const action = tryPlaceCatastrophe(state, player)
    if (action) return action
  }

  // Swap if hand has nothing useful for current on-board leaders.
  if (onBoardLeaders.length > 0 && player.hand.length > 0) {
    const onBoardColors = new Set(onBoardLeaders.map(l => l.color))
    const hasMatchingTile = player.hand.some(t => onBoardColors.has(t))
    if (!hasMatchingTile) {
      const indices = player.hand.map((_, i) => i)
      return { type: 'swapTiles', indices }
    }
  }

  const anyTileAction = tryPlaceAnyTile(state, player)
  if (anyTileAction) return anyTileAction

  // Prefer swap over pass when stuck.
  if (player.hand.length > 0) {
    const indices = player.hand.map((_, i) => i)
    return { type: 'swapTiles', indices }
  }

  return { type: 'pass' }
}

/**
 * Returns the position of a same-color enemy leader in a kingdom that the given
 * placement would join, or null if no revolt would be triggered.
 */
function findRevoltDefender(
  state: GameState,
  color: LeaderColor,
  placementPos: Position,
): Position | null {
  const kingdoms = findKingdoms(state.board)
  const playerLeaderDynasty = state.players[state.currentPlayer].dynasty

  const neighbors = getNeighbors(placementPos)
  for (const kingdom of kingdoms) {
    const posSet = new Set(kingdom.positions.map(p => `${p.row},${p.col}`))
    const isAdjacent = neighbors.some(n => posSet.has(`${n.row},${n.col}`))
    if (!isAdjacent) continue

    const conflictLeader = kingdom.leaders.find(
      l => l.color === color && l.dynasty !== playerLeaderDynasty,
    )
    if (conflictLeader?.position) return conflictLeader.position
  }
  return null
}

interface KingdomLite {
  positions: Position[]
  posKeySet: Set<string>
  leaders: { color: LeaderColor; dynasty: string }[]
  treasureCount: number
  tileCount: number
}

function summarizeKingdoms(state: GameState): KingdomLite[] {
  const kingdoms = findKingdoms(state.board)
  return kingdoms.map(k => {
    const posKeySet = new Set(k.positions.map(p => `${p.row},${p.col}`))
    let treasureCount = 0
    let tileCount = 0
    for (const pos of k.positions) {
      const cell = state.board[pos.row][pos.col]
      if (cell.hasTreasure) treasureCount++
      if (cell.tile !== null) tileCount++
    }
    return {
      positions: k.positions,
      posKeySet,
      leaders: k.leaders.map(l => ({ color: l.color, dynasty: l.dynasty })),
      treasureCount,
      tileCount,
    }
  })
}

function adjacentKingdomFor(pos: Position, kingdoms: KingdomLite[]): KingdomLite | null {
  const neighbors = getNeighborPositions(pos)
  for (const k of kingdoms) {
    if (neighbors.some(n => k.posKeySet.has(`${n.row},${n.col}`))) return k
  }
  return null
}

function tryPlaceLeader(
  state: GameState,
  player: GameState['players'][number],
): GameAction | null {
  const colorCounts: Record<string, number> = {}
  for (const tile of player.hand) {
    colorCounts[tile] = (colorCounts[tile] || 0) + 1
  }

  const offBoardLeaders = player.leaders
    .filter(l => l.position === null)
    .sort((a, b) => (colorCounts[b.color] || 0) - (colorCounts[a.color] || 0))

  const kingdoms = summarizeKingdoms(state)

  for (const leader of offBoardLeaders) {
    const validPlacements = getValidLeaderPlacements(state, leader.color)
    if (validPlacements.length === 0) continue

    // Score each placement. Reject placements that lose a revolt; otherwise
    // prefer kingdoms with treasures, more tiles, more adjacent temples (red
    // strength buffer), and bias green leader strongly toward treasure clusters.
    type Scored = { pos: Position; score: number }
    const scored: Scored[] = []
    for (const pos of validPlacements) {
      const defenderPos = findRevoltDefender(state, leader.color, pos)
      if (defenderPos !== null) {
        const myStrength = countAdjacentTemples(state.board, pos)
        const defenderStrength = countAdjacentTemples(state.board, defenderPos)
        const myRedTiles = player.hand.filter(t => t === 'red').length
        if (myStrength + myRedTiles <= defenderStrength) continue
      }

      const adjKingdom = adjacentKingdomFor(pos, kingdoms)
      const treasureCount = adjKingdom?.treasureCount ?? 0
      const tileCount = adjKingdom?.tileCount ?? 0
      const templeAdj = countAdjacentTemples(state.board, pos)
      const treasureWeight = leader.color === 'green' ? 4 : 2
      const score = treasureWeight * treasureCount + 0.5 * tileCount + templeAdj
      scored.push({ pos, score })
    }

    if (scored.length === 0) continue
    scored.sort((a, b) => b.score - a.score)
    return { type: 'placeLeader', color: leader.color, position: scored[0].pos }
  }

  return null
}

function tryPlaceScoringTile(
  state: GameState,
  player: GameState['players'][number],
): GameAction | null {
  const dynasty = player.dynasty
  const onBoardLeaders = player.leaders.filter(l => l.position !== null)
  const leaderColors = new Set(onBoardLeaders.map(l => l.color))
  const kingdoms = summarizeKingdoms(state)

  // Sort by weakest score first (balance VP across colors).
  const sortedColors = [...leaderColors].sort(
    (a, b) => (player.score[a] || 0) - (player.score[b] || 0),
  )

  for (const color of sortedColors) {
    const tileIndex = player.hand.indexOf(color as TileColor)
    if (tileIndex === -1) continue

    const validPlacements = getValidTilePlacements(state, color as TileColor)
    if (validPlacements.length === 0) continue

    const leaderInfo = onBoardLeaders.find(l => l.color === color)
    if (!leaderInfo?.position) continue

    const leaderKingdom = kingdoms.find(k =>
      k.leaders.some(l => l.dynasty === dynasty && l.color === color),
    )

    if (leaderKingdom) {
      // Rank placements adjacent to my leader's kingdom: prefer ones whose
      // *other* neighbors are also tiles (kingdom growth, monument seed) and
      // any placement that adds to a kingdom holding treasures.
      type Scored = { pos: Position; score: number }
      const adjacent: Scored[] = []
      for (const pos of validPlacements) {
        const neighbors = getNeighborPositions(pos)
        const onLeaderKingdom = neighbors.some(n =>
          leaderKingdom.posKeySet.has(`${n.row},${n.col}`),
        )
        if (!onLeaderKingdom) continue

        let neighborTiles = 0
        let sameColorNeighbors = 0
        for (const n of neighbors) {
          const cell = state.board[n.row][n.col]
          if (cell.tile !== null || cell.leader !== null || cell.monument !== null) {
            neighborTiles++
          }
          if (cell.tile === color) sameColorNeighbors++
        }
        // Strongly favor monument 2x2 seeds (same-color neighbors).
        const score = 4 * sameColorNeighbors + neighborTiles + 0.5 * leaderKingdom.treasureCount
        adjacent.push({ pos, score })
      }

      if (adjacent.length > 0) {
        adjacent.sort((a, b) => b.score - a.score)
        return { type: 'placeTile', color: color as TileColor, position: adjacent[0].pos }
      }
    }

    // Fallback: pick the placement that grows an existing kingdom the most.
    const ranked = rankByConnectivity(state, validPlacements)
    return { type: 'placeTile', color: color as TileColor, position: ranked[0] }
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
    if (placements.length === 0) continue
    const ranked = rankByConnectivity(state, placements)
    return { type: 'placeTile', color, position: ranked[0] }
  }
  return null
}

/**
 * Pick a catastrophe placement that disrupts a strong opponent VP pipeline
 * we can't otherwise contest. Returns null if no high-leverage target.
 */
function tryPlaceCatastrophe(
  state: GameState,
  player: GameState['players'][number],
): GameAction | null {
  const myDynasty = player.dynasty
  const myLeaderColors = new Set(
    player.leaders.filter(l => l.position !== null).map(l => l.color),
  )
  const kingdoms = findKingdoms(state.board)

  type Target = { pos: Position; weight: number }
  let best: Target | null = null

  for (const kingdom of kingdoms) {
    // Find opponent leaders in this kingdom by color.
    const oppLeaderColors = new Set(
      kingdom.leaders
        .filter(l => l.dynasty !== myDynasty)
        .map(l => l.color),
    )
    if (oppLeaderColors.size === 0) continue

    for (const color of oppLeaderColors) {
      // Only target colors we cannot contest by tile placement.
      if (myLeaderColors.has(color)) continue

      // Count face-up tiles of this color in the kingdom.
      const colorPositions: Position[] = []
      for (const pos of kingdom.positions) {
        const cell = state.board[pos.row][pos.col]
        if (cell.tile === color && !cell.tileFlipped) {
          colorPositions.push(pos)
        }
      }
      if (colorPositions.length < CATASTROPHE_MIN_TARGET_TILES) continue

      // Choose a single target: highest-leverage face-up tile that's a legal
      // catastrophe placement (no treasure, no adjacent leader, etc).
      for (const pos of colorPositions) {
        if (!canPlaceCatastrophe(state, pos)) continue
        // Skip cells adjacent to one of MY leaders — destroying my own scoring base.
        const neighbors = getNeighborPositions(pos)
        const adjMyLeader = neighbors.some(n => {
          const c = state.board[n.row][n.col]
          return c.leader !== null && c.leader.dynasty === myDynasty
        })
        if (adjMyLeader) continue

        const weight = colorPositions.length + (color === 'red' ? 1 : 0)
        if (!best || weight > best.weight) {
          best = { pos, weight }
        }
      }
    }
  }

  if (!best) return null
  return { type: 'placeCatastrophe', position: best.pos }
}

/**
 * Rank tile placements by adjacency to existing tiles/leaders. Heads off
 * one-tile islands in remote corners.
 */
function rankByConnectivity(state: GameState, placements: Position[]): Position[] {
  const scored = placements.map(pos => {
    const neighbors = getNeighborPositions(pos)
    let occupied = 0
    let sameColor = 0
    const here = state.board[pos.row][pos.col].tile
    for (const n of neighbors) {
      const cell = state.board[n.row][n.col]
      if (cell.tile !== null || cell.leader !== null || cell.monument !== null) {
        occupied++
      }
      if (here && cell.tile === here) sameColor++
    }
    return { pos, score: 2 * occupied + 4 * sameColor }
  })
  scored.sort((a, b) => b.score - a.score)
  return scored.map(s => s.pos)
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
