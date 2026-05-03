/**
 * Browser-safe observation + action encoding for Tigris & Euphrates.
 *
 * Extracted from server.ts so the same encoder can be used both by the Node
 * bridge (for the Python RL pipeline) and the browser (for in-page ONNX
 * inference). Anything Node-specific (readline, process, stdin/stdout)
 * stays in server.ts; this file imports only from src/engine/.
 */

import {
  GameState,
  GameAction,
  TileColor,
  LeaderColor,
  BOARD_ROWS,
  BOARD_COLS,
} from '../engine/types'
import {
  getValidTilePlacements,
  getValidLeaderPlacements,
  canPlaceCatastrophe,
} from '../engine/validation'

export const COLOR_INDEX: Record<string, number> = { red: 0, blue: 1, green: 2, black: 3 }
export const HAND_MAX = 6
export const BOARD_CHANNELS = 15
export const ACTION_SPACE_SIZE =
  8 * BOARD_ROWS * BOARD_COLS + 4 + BOARD_ROWS * BOARD_COLS + 64 + 1 + 64 + 4 + 6 + 1

// Hierarchical action layout. The flat ACTION_SPACE_SIZE is partitioned into
// 10 contiguous parameter ranges, one per action type. The hierarchical
// policy samples a type first (Cat over 10 logits) then a parameter
// conditional on the type (Cat over per-type slot range). Per-state mask
// sums collapse from up to 1728 to 10 (types) + at most 704 (params),
// usually <50 in either head — entropy regularizer no longer pushes mass
// into invalid types.
export const ACTION_TYPES = [
  'placeTile', 'placeLeader', 'withdrawLeader', 'placeCatastrophe',
  'swapTiles', 'pass', 'commitSupport', 'chooseWarOrder',
  'buildMonument', 'declineMonument',
] as const
export const NUM_ACTION_TYPES = ACTION_TYPES.length // 10
export const TYPE_PARAM_SIZES: number[] = [
  4 * BOARD_ROWS * BOARD_COLS, // placeTile (4 colors × cells)
  4 * BOARD_ROWS * BOARD_COLS, // placeLeader (4 colors × cells)
  4,                            // withdrawLeader (color)
  BOARD_ROWS * BOARD_COLS,      // placeCatastrophe (cell)
  64,                           // swapTiles (6-bit hand mask)
  1,                            // pass
  64,                           // commitSupport (6-bit hand mask)
  4,                            // chooseWarOrder (color)
  6,                            // buildMonument (monument index)
  1,                            // declineMonument
]
export const TYPE_BASES: number[] = (() => {
  const out: number[] = []
  let acc = 0
  for (const s of TYPE_PARAM_SIZES) { out.push(acc); acc += s }
  return out
})()

export function decodeFlatAction(index: number): { typeIdx: number; paramIdx: number } {
  for (let t = NUM_ACTION_TYPES - 1; t >= 0; t--) {
    if (index >= TYPE_BASES[t]) return { typeIdx: t, paramIdx: index - TYPE_BASES[t] }
  }
  return { typeIdx: 0, paramIdx: index }
}

const TILE_COLORS: TileColor[] = ['red', 'blue', 'green', 'black']
const LEADER_COLORS: LeaderColor[] = ['red', 'blue', 'green', 'black']

export interface EncodedAction {
  index: number
  typeIdx: number
  paramIdx: number
  action: GameAction
  label: string
}

function ownerOfLeaderAt(state: GameState, r: number, c: number): number {
  for (let pi = 0; pi < state.players.length; pi++) {
    if (state.players[pi].leaders.some(l =>
      l.position && l.position.row === r && l.position.col === c)) {
      return pi
    }
  }
  return -1
}

export function encodeBoard(state: GameState): number[][][] {
  // Channel layout (matches python/tigphrates_env.py BOARD_CHANNELS):
  // Ch 0-3:   tiles by color (binary)
  // Ch 4-7:   leaders by color; value = 1 + ownerPlayerIndex
  // Ch 8:     monument color1 index (1-4), 0 = no monument
  // Ch 9:     monument color2 index (1-4)
  // Ch 10:    monument owner = 1 + ownerPlayerIndex
  // Ch 11:    catastrophes (binary)
  // Ch 12:    treasures (binary)
  // Ch 13:    river terrain (binary)
  // Ch 14:    flipped tiles (binary)
  const board: number[][][] = Array.from({ length: BOARD_CHANNELS }, () =>
    Array.from({ length: BOARD_ROWS }, () => new Array(BOARD_COLS).fill(0))
  )

  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const cell = state.board[r][c]
      if (cell.tile !== null && !cell.tileFlipped) {
        board[COLOR_INDEX[cell.tile]][r][c] = 1
      }
      if (cell.leader !== null) {
        const ci = COLOR_INDEX[cell.leader.color]
        const owner = ownerOfLeaderAt(state, r, c)
        board[4 + ci][r][c] = owner + 1
      }
      if (cell.monument !== null) {
        const m = state.monuments.find(mm => mm.id === cell.monument)
        if (m) {
          board[8][r][c] = COLOR_INDEX[m.color1] + 1
          board[9][r][c] = COLOR_INDEX[m.color2] + 1
          let owner = -1
          for (let dr = -1; dr <= 1 && owner < 0; dr++) {
            for (let dc = -1; dc <= 1 && owner < 0; dc++) {
              if (dr === 0 && dc === 0) continue
              const nr = r + dr, nc = c + dc
              if (nr < 0 || nr >= BOARD_ROWS || nc < 0 || nc >= BOARD_COLS) continue
              owner = ownerOfLeaderAt(state, nr, nc)
            }
          }
          board[10][r][c] = owner + 1
        }
      }
      if (cell.catastrophe) board[11][r][c] = 1
      if (cell.hasTreasure) board[12][r][c] = 1
      if (cell.terrain === 'river') board[13][r][c] = 1
      if (cell.tileFlipped) board[14][r][c] = 1
    }
  }
  return board
}

export function encodeObservation(state: GameState, playerIndex: number) {
  const player = state.players[playerIndex]
  const hand = [0, 0, 0, 0]
  for (const tile of player.hand) hand[COLOR_INDEX[tile]]++

  const handSeq: number[] = new Array(HAND_MAX).fill(-1)
  for (let i = 0; i < player.hand.length && i < HAND_MAX; i++) {
    handSeq[i] = COLOR_INDEX[player.hand[i]]
  }

  const leaderPositions: number[] = []
  for (const leader of player.leaders) {
    if (leader.position) {
      leaderPositions.push(leader.position.row, leader.position.col)
    } else {
      leaderPositions.push(-1, -1)
    }
  }

  const scores = [player.score.red, player.score.blue, player.score.green, player.score.black]

  const opponentScores: number[][] = []
  const opponentLeaderPositions: number[][] = []
  for (let i = 0; i < state.players.length; i++) {
    if (i === playerIndex) continue
    const opp = state.players[i]
    opponentScores.push([opp.score.red, opp.score.blue, opp.score.green, opp.score.black])
    const oppLeaders: number[] = []
    for (const leader of opp.leaders) {
      if (leader.position) {
        oppLeaders.push(leader.position.row, leader.position.col)
      } else {
        oppLeaders.push(-1, -1)
      }
    }
    opponentLeaderPositions.push(oppLeaders)
  }

  const PHASE_INDEX: Record<string, number> = {
    action: 0, conflictSupport: 1, warOrderChoice: 2, monumentChoice: 3, gameOver: 4,
  }

  return {
    board: encodeBoard(state),
    hand,
    handSeq,
    scores,
    treasures: player.treasures,
    catastrophesRemaining: player.catastrophesRemaining,
    leaderPositions,
    opponentScores,
    opponentLeaderPositions,
    bagSize: state.bag.length,
    actionsRemaining: state.actionsRemaining,
    turnPhase: PHASE_INDEX[state.turnPhase] ?? 0,
    currentPlayer: state.currentPlayer,
    playerIndex,
    numPlayers: state.players.length,
    conflict: state.pendingConflict ? {
      type: state.pendingConflict.type === 'revolt' ? 0 : 1,
      color: COLOR_INDEX[state.pendingConflict.color],
      attackerStrength: state.pendingConflict.attackerStrength,
      defenderStrength: state.pendingConflict.defenderStrength,
      attackerCommitted: state.pendingConflict.attackerCommitted !== null,
      isAttacker: state.pendingConflict.attacker.playerIndex === playerIndex,
      isDefender: state.pendingConflict.defender.playerIndex === playerIndex,
    } : null,
  }
}

export function activePlayerIndex(state: GameState): number {
  if (state.turnPhase === 'conflictSupport') {
    const conflict = state.pendingConflict!
    return conflict.attackerCommitted === null
      ? conflict.attacker.playerIndex
      : conflict.defender.playerIndex
  }
  return state.currentPlayer
}

function makeAction(index: number, action: GameAction, label: string): EncodedAction {
  const { typeIdx, paramIdx } = decodeFlatAction(index)
  return { index, typeIdx, paramIdx, action, label }
}

export function enumerateValidActions(state: GameState): EncodedAction[] {
  const actions: EncodedAction[] = []
  const playerIndex = activePlayerIndex(state)
  const player = state.players[playerIndex]

  if (state.turnPhase === 'action') {
    for (let ci = 0; ci < 4; ci++) {
      const color = TILE_COLORS[ci]
      if (!player.hand.includes(color)) continue
      const placements = getValidTilePlacements(state, color)
      for (const pos of placements) {
        const idx = ci * BOARD_ROWS * BOARD_COLS + pos.row * BOARD_COLS + pos.col
        actions.push(makeAction(
          idx,
          { type: 'placeTile', color, position: pos },
          `placeTile:${color}@${pos.row},${pos.col}`,
        ))
      }
    }

    const BASE_LEADER = 4 * BOARD_ROWS * BOARD_COLS
    for (let ci = 0; ci < 4; ci++) {
      const color = LEADER_COLORS[ci]
      const leader = player.leaders.find(l => l.color === color)
      if (!leader || leader.position !== null) continue
      const placements = getValidLeaderPlacements(state, color)
      for (const pos of placements) {
        const idx = BASE_LEADER + ci * BOARD_ROWS * BOARD_COLS + pos.row * BOARD_COLS + pos.col
        actions.push(makeAction(
          idx,
          { type: 'placeLeader', color, position: pos },
          `placeLeader:${color}@${pos.row},${pos.col}`,
        ))
      }
    }

    const BASE_WITHDRAW = 8 * BOARD_ROWS * BOARD_COLS
    for (let ci = 0; ci < 4; ci++) {
      const color = LEADER_COLORS[ci]
      const leader = player.leaders.find(l => l.color === color)
      if (leader?.position) {
        actions.push(makeAction(
          BASE_WITHDRAW + ci,
          { type: 'withdrawLeader', color },
          `withdraw:${color}`,
        ))
      }
    }

    const BASE_CATASTROPHE = BASE_WITHDRAW + 4
    if (player.catastrophesRemaining > 0) {
      for (let r = 0; r < BOARD_ROWS; r++) {
        for (let c = 0; c < BOARD_COLS; c++) {
          if (canPlaceCatastrophe(state, { row: r, col: c })) {
            actions.push(makeAction(
              BASE_CATASTROPHE + r * BOARD_COLS + c,
              { type: 'placeCatastrophe', position: { row: r, col: c } },
              `catastrophe@${r},${c}`,
            ))
          }
        }
      }
    }

    const BASE_SWAP = BASE_CATASTROPHE + BOARD_ROWS * BOARD_COLS
    if (player.hand.length > 0) {
      const maxMask = (1 << player.hand.length) - 1
      for (let mask = 1; mask <= Math.min(maxMask, 63); mask++) {
        const indices: number[] = []
        for (let b = 0; b < player.hand.length; b++) {
          if (mask & (1 << b)) indices.push(b)
        }
        actions.push(makeAction(
          BASE_SWAP + mask,
          { type: 'swapTiles', indices },
          `swap(${indices.length})`,
        ))
      }
    }

    const BASE_PASS = BASE_SWAP + 64
    actions.push(makeAction(BASE_PASS, { type: 'pass' }, 'pass'))
  } else if (state.turnPhase === 'conflictSupport') {
    const BASE_SUPPORT = 8 * BOARD_ROWS * BOARD_COLS + 4 + BOARD_ROWS * BOARD_COLS + 64 + 1
    const conflict = state.pendingConflict!
    const supportColor: TileColor = conflict.type === 'revolt' ? 'red' : conflict.color as TileColor
    const matchingHandIndices: number[] = []
    for (let i = 0; i < player.hand.length; i++) {
      if (player.hand[i] === supportColor) matchingHandIndices.push(i)
    }
    const subsetCount = 1 << matchingHandIndices.length
    for (let mask = 0; mask < Math.min(subsetCount, 64); mask++) {
      const indices: number[] = []
      for (let b = 0; b < matchingHandIndices.length; b++) {
        if (mask & (1 << b)) indices.push(matchingHandIndices[b])
      }
      actions.push(makeAction(
        BASE_SUPPORT + mask,
        { type: 'commitSupport', indices },
        `commitSupport(${indices.length})`,
      ))
    }
  } else if (state.turnPhase === 'warOrderChoice') {
    const BASE_WAR_ORDER = 8 * BOARD_ROWS * BOARD_COLS + 4 + BOARD_ROWS * BOARD_COLS + 64 + 1 + 64
    const pending = state.pendingConflict?.pendingWarColors ?? []
    for (const color of pending) {
      const ci = COLOR_INDEX[color]
      actions.push(makeAction(
        BASE_WAR_ORDER + ci,
        { type: 'chooseWarOrder', color },
        `warOrder:${color}`,
      ))
    }
  } else if (state.turnPhase === 'monumentChoice') {
    const BASE_MONUMENT = 8 * BOARD_ROWS * BOARD_COLS + 4 + BOARD_ROWS * BOARD_COLS + 64 + 1 + 64 + 4
    const pending = state.pendingMonument
    if (pending) {
      const monuments = state.monuments.filter(
        m => m.position === null && (m.color1 === pending.color || m.color2 === pending.color)
      )
      for (let i = 0; i < monuments.length; i++) {
        const mIdx = state.monuments.indexOf(monuments[i])
        actions.push(makeAction(
          BASE_MONUMENT + mIdx,
          { type: 'buildMonument', monumentId: monuments[i].id },
          `buildMonument:${monuments[i].id}`,
        ))
      }
    }
    actions.push(makeAction(
      BASE_MONUMENT + 6,
      { type: 'declineMonument' },
      'declineMonument',
    ))
  }

  return actions
}

export function createActionMask(actions: EncodedAction[]): number[] {
  const mask = new Array(ACTION_SPACE_SIZE).fill(0)
  for (const a of actions) mask[a.index] = 1
  return mask
}
