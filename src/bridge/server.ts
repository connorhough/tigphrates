/**
 * JSON-RPC-style bridge server for the game engine.
 * Communicates via newline-delimited JSON on stdin/stdout.
 *
 * Protocol:
 *   Request:  { "id": number, "method": string, "params": object }
 *   Response: { "id": number, "result"?: object, "error"?: string }
 *
 * Methods:
 *   create          { playerCount: number }                         -> { gameId }
 *   step            { gameId, action, playerIndex }                 -> { state, reward, done, info }
 *   get_state       { gameId }                                      -> { state }
 *   get_observation { gameId, playerIndex }                         -> { observation }
 *   valid_actions   { gameId }                                      -> { actions }
 *   reset           { gameId, playerCount?: number }                -> { state }
 *   ai_action       { gameId }                                      -> { action }
 */

import * as readline from 'readline'
import { createGame } from '../engine/setup'
import { gameReducer } from '../engine/reducer'
import { GameState, GameAction, TileColor, LeaderColor, BOARD_ROWS, BOARD_COLS } from '../engine/types'
import { getValidTilePlacements, getValidLeaderPlacements, canPlaceCatastrophe } from '../engine/validation'
import { getAIAction } from '../ai/simpleAI'

// --- Game storage ---
const games = new Map<number, GameState>()
let nextGameId = 1

// --- Observation encoding ---

const COLOR_INDEX: Record<string, number> = { red: 0, blue: 1, green: 2, black: 3 }
const DYNASTY_INDEX: Record<string, number> = { archer: 0, bull: 1, pot: 2, lion: 3 }

function encodeBoard(state: GameState): number[][][] {
  // 13 channels × 11 rows × 16 cols
  // Ch 0-3: tiles by color (1 if present, 0 otherwise)
  // Ch 4-7: leaders by dynasty (color encoded as value 1-4)
  // Ch 8: monuments (1 if present)
  // Ch 9: catastrophes (1 if present)
  // Ch 10: treasures (1 if present)
  // Ch 11: terrain (1 = river, 0 = land)
  // Ch 12: flipped tiles (1 if flipped)
  const channels = 13
  const board: number[][][] = Array.from({ length: channels }, () =>
    Array.from({ length: BOARD_ROWS }, () => new Array(BOARD_COLS).fill(0))
  )

  for (let r = 0; r < BOARD_ROWS; r++) {
    for (let c = 0; c < BOARD_COLS; c++) {
      const cell = state.board[r][c]
      if (cell.tile !== null && !cell.tileFlipped) {
        board[COLOR_INDEX[cell.tile]][r][c] = 1
      }
      if (cell.leader !== null) {
        const di = DYNASTY_INDEX[cell.leader.dynasty]
        board[4 + di][r][c] = COLOR_INDEX[cell.leader.color] + 1
      }
      if (cell.monument !== null) board[8][r][c] = 1
      if (cell.catastrophe) board[9][r][c] = 1
      if (cell.hasTreasure) board[10][r][c] = 1
      if (cell.terrain === 'river') board[11][r][c] = 1
      if (cell.tileFlipped) board[12][r][c] = 1
    }
  }
  return board
}

function encodeObservation(state: GameState, playerIndex: number) {
  const player = state.players[playerIndex]
  const hand = [0, 0, 0, 0] // red, blue, green, black
  for (const tile of player.hand) hand[COLOR_INDEX[tile]]++

  const leaderPositions: number[] = []
  for (const leader of player.leaders) {
    if (leader.position) {
      leaderPositions.push(leader.position.row, leader.position.col)
    } else {
      leaderPositions.push(-1, -1)
    }
  }

  const scores = [player.score.red, player.score.blue, player.score.green, player.score.black]

  // Opponent public info
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
    // Conflict info
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

// --- Valid action enumeration ---

interface EncodedAction {
  index: number  // flat action index
  action: GameAction
  label: string
}

const TILE_COLORS: TileColor[] = ['red', 'blue', 'green', 'black']
const LEADER_COLORS: LeaderColor[] = ['red', 'blue', 'green', 'black']

function activePlayerIndex(state: GameState): number {
  if (state.turnPhase === 'conflictSupport') {
    const conflict = state.pendingConflict!
    return conflict.attackerCommitted === null
      ? conflict.attacker.playerIndex
      : conflict.defender.playerIndex
  }
  return state.currentPlayer
}

function enumerateValidActions(state: GameState): EncodedAction[] {
  const actions: EncodedAction[] = []
  const playerIndex = activePlayerIndex(state)
  const player = state.players[playerIndex]

  if (state.turnPhase === 'action') {
    // placeTile: 4 colors × 176 positions = indices 0-703
    for (let ci = 0; ci < 4; ci++) {
      const color = TILE_COLORS[ci]
      if (!player.hand.includes(color)) continue
      const placements = getValidTilePlacements(state, color)
      for (const pos of placements) {
        const idx = ci * BOARD_ROWS * BOARD_COLS + pos.row * BOARD_COLS + pos.col
        actions.push({
          index: idx,
          action: { type: 'placeTile', color, position: pos },
          label: `placeTile:${color}@${pos.row},${pos.col}`,
        })
      }
    }

    // placeLeader: 4 colors × 176 positions = indices 704-1407
    const BASE_LEADER = 4 * BOARD_ROWS * BOARD_COLS
    for (let ci = 0; ci < 4; ci++) {
      const color = LEADER_COLORS[ci]
      const leader = player.leaders.find(l => l.color === color)
      if (!leader || leader.position !== null) continue
      const placements = getValidLeaderPlacements(state, color)
      for (const pos of placements) {
        const idx = BASE_LEADER + ci * BOARD_ROWS * BOARD_COLS + pos.row * BOARD_COLS + pos.col
        actions.push({
          index: idx,
          action: { type: 'placeLeader', color, position: pos },
          label: `placeLeader:${color}@${pos.row},${pos.col}`,
        })
      }
    }

    // withdrawLeader: 4 colors = indices 1408-1411
    const BASE_WITHDRAW = 8 * BOARD_ROWS * BOARD_COLS
    for (let ci = 0; ci < 4; ci++) {
      const color = LEADER_COLORS[ci]
      const leader = player.leaders.find(l => l.color === color)
      if (leader?.position) {
        actions.push({
          index: BASE_WITHDRAW + ci,
          action: { type: 'withdrawLeader', color },
          label: `withdraw:${color}`,
        })
      }
    }

    // placeCatastrophe: 176 positions = indices 1412-1587
    const BASE_CATASTROPHE = BASE_WITHDRAW + 4
    if (player.catastrophesRemaining > 0) {
      for (let r = 0; r < BOARD_ROWS; r++) {
        for (let c = 0; c < BOARD_COLS; c++) {
          if (canPlaceCatastrophe(state, { row: r, col: c })) {
            actions.push({
              index: BASE_CATASTROPHE + r * BOARD_COLS + c,
              action: { type: 'placeCatastrophe', position: { row: r, col: c } },
              label: `catastrophe@${r},${c}`,
            })
          }
        }
      }
    }

    // swapTiles: 64 subsets (6-bit mask) = indices 1588-1651
    // Only non-empty subsets, max 6 tiles
    const BASE_SWAP = BASE_CATASTROPHE + BOARD_ROWS * BOARD_COLS
    if (player.hand.length > 0) {
      const maxMask = (1 << player.hand.length) - 1
      for (let mask = 1; mask <= Math.min(maxMask, 63); mask++) {
        const indices: number[] = []
        for (let b = 0; b < player.hand.length; b++) {
          if (mask & (1 << b)) indices.push(b)
        }
        actions.push({
          index: BASE_SWAP + mask,
          action: { type: 'swapTiles', indices },
          label: `swap(${indices.length})`,
        })
      }
    }

    // pass = index 1652
    const BASE_PASS = BASE_SWAP + 64
    actions.push({
      index: BASE_PASS,
      action: { type: 'pass' },
      label: 'pass',
    })
  } else if (state.turnPhase === 'conflictSupport') {
    // commitSupport: 64 subsets = indices 1653-1716
    // Only tiles matching the conflict color can be committed
    const BASE_SUPPORT = 8 * BOARD_ROWS * BOARD_COLS + 4 + BOARD_ROWS * BOARD_COLS + 64 + 1
    const conflict = state.pendingConflict!
    const supportColor: TileColor = conflict.type === 'revolt' ? 'red' : conflict.color as TileColor
    const matchingHandIndices: number[] = []
    for (let i = 0; i < player.hand.length; i++) {
      if (player.hand[i] === supportColor) matchingHandIndices.push(i)
    }
    // Generate subsets of matching tiles (including empty = commit nothing)
    const subsetCount = 1 << matchingHandIndices.length
    for (let mask = 0; mask < Math.min(subsetCount, 64); mask++) {
      const indices: number[] = []
      for (let b = 0; b < matchingHandIndices.length; b++) {
        if (mask & (1 << b)) indices.push(matchingHandIndices[b])
      }
      actions.push({
        index: BASE_SUPPORT + mask,
        action: { type: 'commitSupport', indices },
        label: `commitSupport(${indices.length})`,
      })
    }
  } else if (state.turnPhase === 'warOrderChoice') {
    // chooseWarOrder: 4 colors = indices 1717-1720
    const BASE_WAR_ORDER = 8 * BOARD_ROWS * BOARD_COLS + 4 + BOARD_ROWS * BOARD_COLS + 64 + 1 + 64
    const pending = state.pendingConflict?.pendingWarColors ?? []
    for (const color of pending) {
      const ci = COLOR_INDEX[color]
      actions.push({
        index: BASE_WAR_ORDER + ci,
        action: { type: 'chooseWarOrder', color },
        label: `warOrder:${color}`,
      })
    }
  } else if (state.turnPhase === 'monumentChoice') {
    // buildMonument: 6 IDs = indices 1721-1726
    const BASE_MONUMENT = 8 * BOARD_ROWS * BOARD_COLS + 4 + BOARD_ROWS * BOARD_COLS + 64 + 1 + 64 + 4
    const pending = state.pendingMonument
    if (pending) {
      const monuments = state.monuments.filter(
        m => m.position === null && (m.color1 === pending.color || m.color2 === pending.color)
      )
      for (let i = 0; i < monuments.length; i++) {
        const mIdx = state.monuments.indexOf(monuments[i])
        actions.push({
          index: BASE_MONUMENT + mIdx,
          action: { type: 'buildMonument', monumentId: monuments[i].id },
          label: `buildMonument:${monuments[i].id}`,
        })
      }
    }
    // declineMonument = index 1727
    actions.push({
      index: BASE_MONUMENT + 6,
      action: { type: 'declineMonument' },
      label: 'declineMonument',
    })
  }

  return actions
}

// Total action space size
const ACTION_SPACE_SIZE = 8 * BOARD_ROWS * BOARD_COLS + 4 + BOARD_ROWS * BOARD_COLS + 64 + 1 + 64 + 4 + 6 + 1

// --- Reward computation ---

function computeReward(
  prevState: GameState,
  nextState: GameState,
  playerIndex: number,
): number {
  if (nextState.turnPhase === 'gameOver') {
    // Terminal: +1 for winner, -1 for others
    const scores = nextState.players.map(p =>
      Math.min(p.score.red, p.score.blue, p.score.green, p.score.black) + p.treasures
    )
    const maxScore = Math.max(...scores)
    return scores[playerIndex] === maxScore ? 1.0 : -1.0
  }

  // Intermediate shaping: delta of min-score (small)
  const prevMin = Math.min(
    prevState.players[playerIndex].score.red,
    prevState.players[playerIndex].score.blue,
    prevState.players[playerIndex].score.green,
    prevState.players[playerIndex].score.black,
  )
  const nextMin = Math.min(
    nextState.players[playerIndex].score.red,
    nextState.players[playerIndex].score.blue,
    nextState.players[playerIndex].score.green,
    nextState.players[playerIndex].score.black,
  )
  return (nextMin - prevMin) * 0.01
}

// --- Request handlers ---

function handleCreate(params: { playerCount: number }) {
  const { playerCount } = params
  const aiFlags = new Array(playerCount).fill(true)
  const state = createGame(playerCount, aiFlags)
  const id = nextGameId++
  games.set(id, state)
  return { gameId: id, actionSpaceSize: ACTION_SPACE_SIZE }
}

function handleStep(params: { gameId: number; actionIndex: number; playerIndex: number }) {
  const { gameId, actionIndex, playerIndex } = params
  const state = games.get(gameId)
  if (!state) throw new Error(`Unknown game ${gameId}`)

  // Find the action matching this index
  const validActions = enumerateValidActions(state)
  const matched = validActions.find(a => a.index === actionIndex)
  if (!matched) {
    throw new Error(`Invalid action index ${actionIndex}. Valid: [${validActions.map(a => a.index).join(',')}]`)
  }

  const prevState = state
  const nextState = gameReducer(state, { ...matched.action, playerIndex })
  games.set(gameId, nextState)

  const reward = computeReward(prevState, nextState, playerIndex)
  const done = nextState.turnPhase === 'gameOver'

  return {
    reward,
    done,
    activePlayer: activePlayerIndex(nextState),
    turnPhase: nextState.turnPhase,
    info: done ? {
      scores: nextState.players.map(p => ({
        min: Math.min(p.score.red, p.score.blue, p.score.green, p.score.black),
        red: p.score.red, blue: p.score.blue, green: p.score.green, black: p.score.black,
        treasures: p.treasures,
      })),
    } : undefined,
  }
}

function handleStepAction(params: { gameId: number; action: GameAction; playerIndex: number }) {
  const { gameId, action, playerIndex } = params
  const state = games.get(gameId)
  if (!state) throw new Error(`Unknown game ${gameId}`)

  const prevState = state
  const nextState = gameReducer(state, { ...action, playerIndex })
  games.set(gameId, nextState)

  const reward = computeReward(prevState, nextState, playerIndex)
  const done = nextState.turnPhase === 'gameOver'

  return {
    reward,
    done,
    activePlayer: activePlayerIndex(nextState),
    turnPhase: nextState.turnPhase,
  }
}

function handleGetState(params: { gameId: number }) {
  const state = games.get(params.gameId)
  if (!state) throw new Error(`Unknown game ${params.gameId}`)
  return { state }
}

function handleGetObservation(params: { gameId: number; playerIndex: number }) {
  const state = games.get(params.gameId)
  if (!state) throw new Error(`Unknown game ${params.gameId}`)
  return encodeObservation(state, params.playerIndex)
}

function handleValidActions(params: { gameId: number }) {
  const state = games.get(params.gameId)
  if (!state) throw new Error(`Unknown game ${params.gameId}`)
  const actions = enumerateValidActions(state)
  return {
    activePlayer: activePlayerIndex(state),
    turnPhase: state.turnPhase,
    actions: actions.map(a => ({ index: a.index, label: a.label })),
    mask: createActionMask(actions),
  }
}

function createActionMask(actions: EncodedAction[]): number[] {
  const mask = new Array(ACTION_SPACE_SIZE).fill(0)
  for (const a of actions) mask[a.index] = 1
  return mask
}

function handleReset(params: { gameId: number; playerCount?: number }) {
  const state = games.get(params.gameId)
  const pc = params.playerCount ?? state?.players.length ?? 2
  const aiFlags = new Array(pc).fill(true)
  const newState = createGame(pc, aiFlags)
  games.set(params.gameId, newState)
  return { gameId: params.gameId }
}

function handleAIAction(params: { gameId: number }) {
  const state = games.get(params.gameId)
  if (!state) throw new Error(`Unknown game ${params.gameId}`)
  const action = getAIAction(state)
  // Find the matching index
  const validActions = enumerateValidActions(state)
  const matched = validActions.find(a =>
    JSON.stringify(a.action) === JSON.stringify(action)
  )
  return {
    action,
    actionIndex: matched?.index ?? -1,
    label: matched?.label ?? 'unknown',
  }
}

// --- Main loop ---

const HANDLERS: Record<string, (params: any) => any> = {
  create: handleCreate,
  step: handleStep,
  step_action: handleStepAction,
  get_state: handleGetState,
  get_observation: handleGetObservation,
  valid_actions: handleValidActions,
  reset: handleReset,
  ai_action: handleAIAction,
}

const rl = readline.createInterface({ input: process.stdin })

rl.on('line', (line: string) => {
  let id = 0
  try {
    const req = JSON.parse(line)
    id = req.id ?? 0
    const handler = HANDLERS[req.method]
    if (!handler) throw new Error(`Unknown method: ${req.method}`)
    const result = handler(req.params ?? {})
    process.stdout.write(JSON.stringify({ id, result }) + '\n')
  } catch (err: any) {
    process.stdout.write(JSON.stringify({ id, error: err.message ?? String(err) }) + '\n')
  }
})

// Signal readiness
process.stdout.write(JSON.stringify({ id: 0, result: { ready: true, actionSpaceSize: ACTION_SPACE_SIZE } }) + '\n')
