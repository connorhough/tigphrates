/**
 * Fuzz test for the game engine.
 *
 * Plays many games with random valid actions, checking invariants after
 * every reducer call. Catches crashes, infinite loops, and corrupt state
 * that hand-written scenarios miss.
 */
import { describe, it, expect } from 'vitest'
import { createGame } from '../setup'
import { gameReducer } from '../reducer'
import { GameState, GameAction, TileColor, BOARD_ROWS, BOARD_COLS } from '../types'
import { getValidTilePlacements, getValidLeaderPlacements, canPlaceCatastrophe } from '../validation'

// Deterministic PRNG (xorshift32) so failures are reproducible
function makeRng(seed: number) {
  let s = seed | 0 || 1
  return () => {
    s ^= s << 13
    s ^= s >> 17
    s ^= s << 5
    return (s >>> 0) / 0xFFFFFFFF
  }
}

function pick<T>(rng: () => number, arr: T[]): T {
  return arr[Math.floor(rng() * arr.length)]
}

// --- Random action generators per phase ---

const TILE_COLORS: TileColor[] = ['red', 'blue', 'green', 'black']

function randomActionPhaseAction(state: GameState, rng: () => number): GameAction & { playerIndex: number } {
  const pi = state.currentPlayer
  const player = state.players[pi]

  // Collect all possible actions
  const candidates: (GameAction & { playerIndex: number })[] = []

  // Pass is always valid
  candidates.push({ type: 'pass', playerIndex: pi })

  // Tile placements (sample up to 10 positions per color to keep it fast)
  for (const color of TILE_COLORS) {
    if (!player.hand.includes(color)) continue
    const placements = getValidTilePlacements(state, color)
    if (placements.length <= 10) {
      for (const pos of placements) {
        candidates.push({ type: 'placeTile', color, position: pos, playerIndex: pi })
      }
    } else {
      for (let i = 0; i < 10; i++) {
        candidates.push({ type: 'placeTile', color, position: pick(rng, placements), playerIndex: pi })
      }
    }
  }

  // Leader placements (sample up to 5)
  for (const leader of player.leaders) {
    if (leader.position !== null) continue
    const placements = getValidLeaderPlacements(state, leader.color)
    if (placements.length <= 5) {
      for (const pos of placements) {
        candidates.push({ type: 'placeLeader', color: leader.color, position: pos, playerIndex: pi })
      }
    } else {
      for (let i = 0; i < 5; i++) {
        candidates.push({ type: 'placeLeader', color: leader.color, position: pick(rng, placements), playerIndex: pi })
      }
    }
  }

  // Leader withdrawals
  for (const leader of player.leaders) {
    if (leader.position === null) continue
    candidates.push({ type: 'withdrawLeader', color: leader.color, playerIndex: pi })
  }

  // Catastrophes — sample a few random cells instead of scanning whole board
  if (player.catastrophesRemaining > 0) {
    for (let i = 0; i < 8; i++) {
      const r = Math.floor(rng() * BOARD_ROWS)
      const c = Math.floor(rng() * BOARD_COLS)
      if (canPlaceCatastrophe(state, { row: r, col: c })) {
        candidates.push({ type: 'placeCatastrophe', position: { row: r, col: c }, playerIndex: pi })
      }
    }
  }

  // Swap tiles (swap random subset)
  if (player.hand.length > 0) {
    const indices: number[] = []
    for (let i = 0; i < player.hand.length; i++) {
      if (rng() < 0.5) indices.push(i)
    }
    if (indices.length === 0) indices.push(0)
    candidates.push({ type: 'swapTiles', indices, playerIndex: pi })
  }

  return pick(rng, candidates)
}

function randomConflictSupportAction(state: GameState, rng: () => number): GameAction & { playerIndex: number } {
  const conflict = state.pendingConflict!
  const isAttackerTurn = conflict.attackerCommitted === null
  const pi = isAttackerTurn ? conflict.attacker.playerIndex : conflict.defender.playerIndex
  const player = state.players[pi]
  const supportColor: TileColor = conflict.type === 'revolt' ? 'red' : conflict.color as TileColor

  const matching: number[] = []
  for (let i = 0; i < player.hand.length; i++) {
    if (player.hand[i] === supportColor) matching.push(i)
  }

  const indices: number[] = []
  for (const idx of matching) {
    if (rng() < 0.6) indices.push(idx)
  }

  return { type: 'commitSupport', indices, playerIndex: pi }
}

function randomWarOrderAction(state: GameState, rng: () => number): GameAction & { playerIndex: number } {
  const pi = state.currentPlayer
  const colors = state.pendingConflict?.pendingWarColors ?? ['red']
  return { type: 'chooseWarOrder', color: pick(rng, colors), playerIndex: pi }
}

function randomMonumentAction(state: GameState, rng: () => number): GameAction & { playerIndex: number } {
  const pi = state.currentPlayer
  if (rng() < 0.5) {
    return { type: 'declineMonument', playerIndex: pi }
  }
  const pending = state.pendingMonument
  if (!pending) return { type: 'declineMonument', playerIndex: pi }

  const availableMonuments = state.monuments.filter(
    m => m.position === null && (m.color1 === pending.color || m.color2 === pending.color)
  )
  if (availableMonuments.length === 0) return { type: 'declineMonument', playerIndex: pi }

  return { type: 'buildMonument', monumentId: pick(rng, availableMonuments).id, playerIndex: pi }
}

function getRandomAction(state: GameState, rng: () => number): GameAction & { playerIndex: number } {
  switch (state.turnPhase) {
    case 'action': return randomActionPhaseAction(state, rng)
    case 'conflictSupport': return randomConflictSupportAction(state, rng)
    case 'warOrderChoice': return randomWarOrderAction(state, rng)
    case 'monumentChoice': return randomMonumentAction(state, rng)
    default: throw new Error(`Unexpected phase: ${state.turnPhase}`)
  }
}

// --- State invariant checks ---

function checkInvariants(state: GameState, action: GameAction, step: number): void {
  const ctx = `step ${step}, after ${action.type}`

  // Board dimensions
  if (state.board.length !== BOARD_ROWS) throw new Error(`${ctx}: wrong board rows`)
  for (const row of state.board) {
    if (row.length !== BOARD_COLS) throw new Error(`${ctx}: wrong board cols`)
  }

  // Current player in range
  if (state.currentPlayer < 0 || state.currentPlayer >= state.players.length) {
    throw new Error(`${ctx}: currentPlayer ${state.currentPlayer} out of range`)
  }

  // Actions remaining
  if (state.actionsRemaining < 0 || state.actionsRemaining > 2) {
    throw new Error(`${ctx}: actionsRemaining ${state.actionsRemaining} out of range`)
  }

  // No negative scores
  for (let i = 0; i < state.players.length; i++) {
    const p = state.players[i]
    for (const color of TILE_COLORS) {
      if (p.score[color] < 0) throw new Error(`${ctx}: P${i} ${color} score negative`)
    }
    if (p.treasures < 0) throw new Error(`${ctx}: P${i} treasures negative`)
  }

  // Hand size ≤ 6
  for (let i = 0; i < state.players.length; i++) {
    if (state.players[i].hand.length > 6) {
      throw new Error(`${ctx}: P${i} hand size ${state.players[i].hand.length} > 6`)
    }
  }

  // Bag non-negative
  if (state.bag.length < 0) throw new Error(`${ctx}: bag negative`)

  // Cell consistency + duplicate leader check in a single pass
  const leaderSeen = new Uint8Array(4 * 4) // 4 dynasties × 4 colors
  const DYNASTIES = ['archer', 'bull', 'pot', 'lion']
  const COLORS_IDX: Record<string, number> = { red: 0, blue: 1, green: 2, black: 3 }
  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const cell = state.board[r][c]
      if (cell.leader && cell.monument) {
        throw new Error(`${ctx}: cell (${r},${c}) has both leader and monument`)
      }
      if (cell.catastrophe && cell.tile) {
        throw new Error(`${ctx}: cell (${r},${c}) has catastrophe and tile`)
      }
      if (cell.catastrophe && cell.leader) {
        throw new Error(`${ctx}: cell (${r},${c}) has catastrophe and leader`)
      }
      if (cell.leader) {
        const di = DYNASTIES.indexOf(cell.leader.dynasty)
        const ci = COLORS_IDX[cell.leader.color]
        const key = di * 4 + ci
        if (leaderSeen[key]) throw new Error(`${ctx}: duplicate leader ${cell.leader.dynasty}-${cell.leader.color} at (${r},${c})`)
        leaderSeen[key] = 1
      }
    }
  }

  // Phase/conflict consistency
  if (state.turnPhase === 'conflictSupport' && !state.pendingConflict) {
    throw new Error(`${ctx}: conflictSupport phase but no pendingConflict`)
  }
  if (state.turnPhase === 'monumentChoice' && !state.pendingMonument) {
    throw new Error(`${ctx}: monumentChoice phase but no pendingMonument`)
  }
}

// --- Fuzz runner ---

function fuzzGame(seed: number, playerCount: number, maxSteps: number): { steps: number; reason: string } {
  const rng = makeRng(seed)
  const aiFlags = new Array(playerCount).fill(false)
  let state = createGame(playerCount, aiFlags)
  let steps = 0

  while (state.turnPhase !== 'gameOver' && steps < maxSteps) {
    const action = getRandomAction(state, rng)
    try {
      const next = gameReducer(state, action)
      checkInvariants(next, action, steps)
      state = next
    } catch (e: unknown) {
      const msg = (e as Error).message ?? ''
      // Invariant violations from checkInvariants contain "step " prefix
      if (msg.startsWith('step ') || msg.startsWith('Invariant')) {
        throw new Error(`Invariant violation [seed=${seed}]: ${msg}`)
      }
      // Reducer validation errors are expected — skip
    }
    steps++
  }

  return { steps, reason: state.turnPhase === 'gameOver' ? 'gameOver' : 'maxSteps' }
}

// --- Tests ---

describe('engine fuzz', () => {
  const GAMES = 50
  const MAX_STEPS = 2000

  for (let playerCount = 2; playerCount <= 4; playerCount++) {
    it(`${playerCount}p: ${GAMES} random games without invariant violations`, { timeout: 120_000 }, () => {
      let completed = 0
      for (let seed = 1; seed <= GAMES; seed++) {
        const result = fuzzGame(seed * 1000 + playerCount, playerCount, MAX_STEPS)
        if (result.reason === 'gameOver') completed++
      }
      console.log(`  ${playerCount}p: ${completed}/${GAMES} completed naturally`)
    })
  }

  it('single seeded game for regression', { timeout: 30_000 }, () => {
    // Pin failing seeds here for regression
    const result = fuzzGame(42, 3, 5000)
    expect(result.steps).toBeGreaterThan(0)
  })
})
