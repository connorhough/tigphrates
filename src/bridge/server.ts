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
import { GameState, GameAction } from '../engine/types'
import { getAIAction } from '../ai/simpleAI'
import {
  encodeObservation,
  enumerateValidActions,
  createActionMask,
  activePlayerIndex,
  ACTION_SPACE_SIZE,
} from './encoder'

// --- Game storage ---
const games = new Map<number, GameState>()
let nextGameId = 1

// --- Reward computation ---

function computeReward(
  prevState: GameState,
  nextState: GameState,
  playerIndex: number,
): number {
  // Terminal: +1 for winner, -1 for others. All shaping is computed in Python
  // from observation deltas; the bridge no longer adds a Δmin-score signal so
  // the reward is not double-counted.
  if (nextState.turnPhase === 'gameOver') {
    const scores = nextState.players.map(p =>
      Math.min(p.score.red, p.score.blue, p.score.green, p.score.black) + p.treasures
    )
    const maxScore = Math.max(...scores)
    return scores[playerIndex] === maxScore ? 1.0 : -1.0
  }
  // Suppress prevState signal — Python-side shaping owns intermediate rewards.
  void prevState
  return 0
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

/**
 * Load a serialized GameState into the bridge as a new game. Trusts the
 * caller — the state is assumed to come from the same engine version.
 * Returns the new gameId so subsequent get_observation / decode_action calls
 * can target it.
 */
function handleLoadState(params: { state: GameState }) {
  const id = nextGameId++
  games.set(id, params.state)
  return { gameId: id }
}

/**
 * Combined "agent step + run opponents" RPC. Applies the agent's action,
 * then drives the simpleAI heuristic for every non-agent seat until control
 * returns to the agent or the game ends. Returns the post-loop observation
 * and mask in the same call so the Python rollout collapses ~3-6 RPCs per
 * step into 1.
 *
 * Use this fast path only when the opponent policy is the heuristic — for
 * trained-policy opponents Python still needs to insert its own model
 * inference between server-side game steps.
 */
function handleAgentStep(params: {
  gameId: number
  actionIndex: number
  agentPlayer: number
}) {
  const state = games.get(params.gameId)
  if (!state) throw new Error(`Unknown game ${params.gameId}`)

  // 1. Apply the agent's chosen action.
  const agentValid = enumerateValidActions(state)
  const matched = agentValid.find(a => a.index === params.actionIndex)
  if (!matched) {
    throw new Error(`Invalid action index ${params.actionIndex}`)
  }
  const prevForAgent = state
  let next = gameReducer(state, { ...matched.action, playerIndex: params.agentPlayer })
  let reward = computeReward(prevForAgent, next, params.agentPlayer)
  let done = next.turnPhase === 'gameOver'
  let lastBeforeOver = next

  // 2. Drive the heuristic AI until control returns to the agent or game ends.
  let safety = 0
  while (!done && safety < 5000) {
    safety++
    const active = activePlayerIndex(next)
    if (active === params.agentPlayer) break
    const aiAction = getAIAction(next)
    lastBeforeOver = next
    next = gameReducer(next, { ...aiAction, playerIndex: active })
    if (next.turnPhase === 'gameOver') {
      done = true
      reward = computeReward(lastBeforeOver, next, params.agentPlayer)
      break
    }
  }

  games.set(params.gameId, next)

  // 3. Return everything Python needs for the next iteration.
  const valid = enumerateValidActions(next)
  const obs = encodeObservation(next, params.agentPlayer)
  return {
    reward,
    done,
    activePlayer: activePlayerIndex(next),
    turnPhase: next.turnPhase,
    obs,
    mask: createActionMask(valid),
    info: done ? {
      scores: next.players.map(p => ({
        min: Math.min(p.score.red, p.score.blue, p.score.green, p.score.black),
        red: p.score.red, blue: p.score.blue, green: p.score.green, black: p.score.black,
        treasures: p.treasures,
      })),
    } : undefined,
  }
}

/**
 * Discard a game from the in-memory map. External callers (e.g. the Python
 * policy server) should call this when they're done with a game loaded via
 * load_state to keep memory bounded.
 */
function handleDeleteGame(params: { gameId: number }) {
  const had = games.delete(params.gameId)
  return { deleted: had }
}

/**
 * Map a flat action index to a concrete GameAction for the given game.
 * Used by external policy servers to resolve a model's argmax/sample output
 * back into a dispatch-ready action object.
 */
function handleDecodeAction(params: { gameId: number; actionIndex: number }) {
  const state = games.get(params.gameId)
  if (!state) throw new Error(`Unknown game ${params.gameId}`)
  const valid = enumerateValidActions(state)
  const matched = valid.find(a => a.index === params.actionIndex)
  if (!matched) throw new Error(`Invalid action index ${params.actionIndex}`)
  return {
    action: matched.action,
    label: matched.label,
    activePlayer: activePlayerIndex(state),
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
  load_state: handleLoadState,
  decode_action: handleDecodeAction,
  delete_game: handleDeleteGame,
  agent_step: handleAgentStep,
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
