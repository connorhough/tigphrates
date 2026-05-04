import { GameState, GameAction, TileColor, LeaderColor, Position } from '../engine/types'
import { getValidTilePlacements, getValidLeaderPlacements, canPlaceCatastrophe } from '../engine/validation'
import { findKingdoms, getNeighbors, findConnectedGroup } from '../engine/board'
import { countAdjacentTemples, countWarSupportTiles } from '../engine/conflict'

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

  // Catastrophe: fire only when one of the three canonical patterns triggers
  // (leader-sever, kingdom-split, monument-block). See findCatastropheMove.
  if (player.catastrophesRemaining > 0) {
    const action = findCatastropheMove(state, state.currentPlayer)
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
 * Pick a catastrophe placement matching one of three canonical T&E patterns:
 *   1) Sever an opponent leader from its supporting red temples.
 *   2) Split a large opponent kingdom at an articulation point.
 *   3) Block a 3-of-4 monument formation.
 * Patterns are tried in priority order. Returns null if no pattern fires.
 */
export function findCatastropheMove(
  state: GameState,
  playerIndex: number,
): GameAction | null {
  const player = state.players[playerIndex]
  if (player.catastrophesRemaining <= 0) return null

  return (
    findSeverLeaderCatastrophe(state, playerIndex) ??
    findKingdomSplitCatastrophe(state, playerIndex) ??
    findMonumentBlockCatastrophe(state, playerIndex)
  )
}

/**
 * Pattern 1: catastrophe a red temple to strand an opponent leader.
 *
 * Trigger: opponent leader sits in a kingdom with one or more adjacent
 * face-up red temples. Catastrophing a temple that leaves the leader with
 * zero adjacent face-up red temples will force its withdrawal.
 *
 * Guards:
 *  - Opponent must be ahead of the AI in the leader's color score by **>=2**
 *    (a one-point lead is not worth burning a catastrophe over, and the lax
 *    >0 trigger fires too eagerly once the AI falls behind in red).
 *  - The leader's kingdom must contain at least one treasure tile (high-
 *    impact targets only — kingdoms without treasures contribute marginal VP).
 *  - The catastrophe target must NOT be adjacent to one of the AI's own
 *    leaders (so we don't destroy our own support).
 */
function findSeverLeaderCatastrophe(
  state: GameState,
  playerIndex: number,
): GameAction | null {
  const myDynasty = state.players[playerIndex].dynasty
  const myScore = state.players[playerIndex].score
  const kingdoms = findKingdoms(state.board)

  for (const kingdom of kingdoms) {
    const oppLeaders = kingdom.leaders.filter(l => l.dynasty !== myDynasty)
    if (oppLeaders.length === 0) continue

    // Guard (b): kingdom must hold a treasure for severing to be worthwhile.
    let kingdomHasTreasure = false
    for (const pos of kingdom.positions) {
      if (state.board[pos.row][pos.col].hasTreasure) {
        kingdomHasTreasure = true
        break
      }
    }
    if (!kingdomHasTreasure) continue

    for (const leader of oppLeaders) {
      const oppPlayer = state.players.find(
        p => p.dynasty === leader.dynasty,
      )
      if (!oppPlayer) continue
      const scoreColor: TileColor = leader.color as TileColor
      // Guard (a): opponent must be ahead by >=2 in this color.
      if (oppPlayer.score[scoreColor] - myScore[scoreColor] < 2) continue

      // Find adjacent face-up red temples to this leader.
      const adjTemples: Position[] = []
      for (const n of getNeighborPositions(leader.position)) {
        const c = state.board[n.row][n.col]
        if (c.tile === 'red' && !c.tileFlipped) {
          adjTemples.push(n)
        }
      }
      if (adjTemples.length === 0) continue

      for (const target of adjTemples) {
        if (!canPlaceCatastrophe(state, target)) continue
        if (isAdjacentToOwnLeader(state, target, myDynasty)) continue
        // Would catastrophing this leave the leader with zero face-up temples?
        const remaining = adjTemples.filter(
          p => !(p.row === target.row && p.col === target.col),
        )
        if (remaining.length === 0) {
          return { type: 'placeCatastrophe', position: target }
        }
      }
    }
  }

  return null
}

/**
 * Pattern 2: split a large opponent kingdom at an articulation point.
 *
 * Trigger: kingdom has ≥6 occupied cells, contains ≥2 opponent leaders of
 * different colors, and there is a single tile (cell) whose removal would
 * disconnect the kingdom into multiple components.
 *
 * Implementation: brute-force articulation check by removing each candidate
 * tile, running findConnectedGroup from a remaining cell, and comparing the
 * resulting group size to (kingdom.size - 1). The board is 11x16 — fine.
 */
function findKingdomSplitCatastrophe(
  state: GameState,
  playerIndex: number,
): GameAction | null {
  const myDynasty = state.players[playerIndex].dynasty
  const kingdoms = findKingdoms(state.board)

  for (const kingdom of kingdoms) {
    if (kingdom.positions.length < 6) continue

    const oppLeaders = kingdom.leaders.filter(l => l.dynasty !== myDynasty)
    if (oppLeaders.length < 2) continue
    const oppColors = new Set(oppLeaders.map(l => l.color))
    if (oppColors.size < 2) continue

    // Candidate tiles: cells in the kingdom that have a face-up tile and no
    // leader/treasure/monument (i.e., legal catastrophe targets).
    for (const pos of kingdom.positions) {
      const cell = state.board[pos.row][pos.col]
      if (cell.tile === null || cell.tileFlipped) continue
      if (!canPlaceCatastrophe(state, pos)) continue
      if (isAdjacentToOwnLeader(state, pos, myDynasty)) continue

      if (isArticulationPoint(state, kingdom.positions, pos)) {
        return { type: 'placeCatastrophe', position: pos }
      }
    }
  }

  return null
}

/**
 * Pattern 3: catastrophe one of three same-color tiles that form an L-shape
 * inside a 2x2 footprint, preventing the opponent from completing the 2x2
 * monument square next turn.
 *
 * Guards:
 *  - The L-shape must be inside an opponent's kingdom (we're blocking THEIR
 *    monument, not our own).
 *  - Opponent must be ahead of the AI in the relevant color.
 *  - The 4th cell of the 2x2 must be a legal tile placement (i.e. they could
 *    actually complete it next turn).
 */
function findMonumentBlockCatastrophe(
  state: GameState,
  playerIndex: number,
): GameAction | null {
  const myDynasty = state.players[playerIndex].dynasty
  const myScore = state.players[playerIndex].score
  const board = state.board
  const rows = board.length
  const cols = board[0].length

  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const corners: Position[] = [
        { row: r, col: c },
        { row: r, col: c + 1 },
        { row: r + 1, col: c },
        { row: r + 1, col: c + 1 },
      ]
      const cells = corners.map(p => board[p.row][p.col])

      // Find color appearing exactly 3 times across the 2x2 (face-up, same color).
      const colorCounts = new Map<TileColor, number>()
      for (const cell of cells) {
        if (cell.tile && !cell.tileFlipped) {
          colorCounts.set(cell.tile, (colorCounts.get(cell.tile) ?? 0) + 1)
        }
      }
      let threeColor: TileColor | null = null
      for (const [color, count] of colorCounts) {
        if (count === 3) {
          threeColor = color
          break
        }
      }
      if (!threeColor) continue

      // The empty corner must be a legal placement of `threeColor` for the
      // opponent to be one move away from completing the 2x2.
      const emptyCorner = corners.find((p, i) => {
        const cell = cells[i]
        return cell.tile === null && cell.leader === null && !cell.catastrophe && cell.monument === null
      })
      if (!emptyCorner) continue
      const validForColor = getValidTilePlacements(state, threeColor)
      const emptyIsValid = validForColor.some(
        p => p.row === emptyCorner.row && p.col === emptyCorner.col,
      )
      if (!emptyIsValid) continue

      // The 2x2 must overlap an opponent kingdom (so they're the ones building it).
      const kingdoms = findKingdoms(state.board)
      const overlappingKingdom = kingdoms.find(k =>
        k.positions.some(p =>
          corners.some(cp => cp.row === p.row && cp.col === p.col),
        ),
      )
      if (!overlappingKingdom) continue
      const oppLeaderInK = overlappingKingdom.leaders.find(l => l.dynasty !== myDynasty)
      if (!oppLeaderInK) continue
      const oppPlayer = state.players.find(p => p.dynasty === oppLeaderInK.dynasty)
      if (!oppPlayer) continue
      // Asymmetric guard.
      if (oppPlayer.score[threeColor] <= myScore[threeColor]) continue

      // Pick any of the 3 same-color corners that's a legal catastrophe target
      // and not adjacent to one of our own leaders.
      for (let i = 0; i < corners.length; i++) {
        const cell = cells[i]
        if (cell.tile !== threeColor || cell.tileFlipped) continue
        if (!canPlaceCatastrophe(state, corners[i])) continue
        if (isAdjacentToOwnLeader(state, corners[i], myDynasty)) continue
        return { type: 'placeCatastrophe', position: corners[i] }
      }
    }
  }

  return null
}

/**
 * Returns true if removing `removed` from the kingdom would disconnect it.
 * Uses brute-force: temporarily clears the cell, walks the connected group
 * from another kingdom cell, compares to expected size.
 */
function isArticulationPoint(
  state: GameState,
  kingdomPositions: Position[],
  removed: Position,
): boolean {
  if (kingdomPositions.length <= 2) return false
  // Pick a starting cell that isn't `removed`.
  const start = kingdomPositions.find(
    p => !(p.row === removed.row && p.col === removed.col),
  )
  if (!start) return false

  // Clone the relevant cell, clear it, run BFS, then restore.
  const cell = state.board[removed.row][removed.col]
  const savedTile = cell.tile
  const savedLeader = cell.leader
  cell.tile = null
  cell.leader = null
  try {
    const group = findConnectedGroup(state.board, start)
    return group.length < kingdomPositions.length - 1
  } finally {
    cell.tile = savedTile
    cell.leader = savedLeader
  }
}

function isAdjacentToOwnLeader(
  state: GameState,
  pos: Position,
  myDynasty: string,
): boolean {
  for (const n of getNeighborPositions(pos)) {
    const c = state.board[n.row][n.col]
    if (c.leader !== null && c.leader.dynasty === myDynasty) return true
  }
  return false
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
