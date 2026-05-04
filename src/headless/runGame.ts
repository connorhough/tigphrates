import { GameState, GameAction, TurnPhase } from '../engine/types'
import { createGame } from '../engine/setup'
import { gameReducer } from '../engine/reducer'
import { getAIAction } from '../ai/simpleAI'
import { getHeadlessOnnxAction } from './onnxAdapter'
import {
  handSummary, scoreSnapshot, formatActionLine, conflictResolutionLine,
} from '../engine/logFormat'

export type AiKind = 'simple' | 'onnx'
export type GetActionFn = (state: GameState, playerIndex: number) => GameAction | Promise<GameAction>

function makeGetAction(kinds: AiKind[], onnxModelPath: string): GetActionFn {
  return async (state, playerIndex) => {
    const kind = kinds[playerIndex] ?? 'simple'
    if (kind === 'onnx') {
      try {
        return await getHeadlessOnnxAction(state, onnxModelPath)
      } catch (e) {
        // ONNX inference failed (model missing, shape mismatch, etc.) —
        // fall back to heuristic so the tournament can still complete.
        // Logged once so we know it happened.
        if (!_warnedOnnxFallback) {
          console.warn(`[onnx] inference failed, falling back to simpleAI:`, e)
          _warnedOnnxFallback = true
        }
        return getAIAction(state)
      }
    }
    return getAIAction(state)
  }
}

let _warnedOnnxFallback = false

// Compact log notation lives in src/engine/logFormat.ts (shared with the
// browser). See that module for the format spec.

// --- Core types ---

export interface GameResult {
  winner: number
  scores: { dynasty: string; minScore: number; score: Record<string, number>; treasures: number }[]
  turnCount: number
  turns: number
  reason: 'gameOver' | 'maxTurns' | 'turnLimit'
  log: string[]
}

export interface RunOptions {
  playerCount: number
  maxActions?: number
  maxTurns?: number
  log?: boolean
  /** Per-player AI kinds. Length must match playerCount. Defaults to all 'simple'. */
  aiKinds?: AiKind[]
  /** Path to ONNX model (used when any aiKind is 'onnx'). */
  onnxModelPath?: string
}

const DEFAULT_MAX_ACTIONS = 5000
const DEFAULT_ONNX_MODEL = 'models/policy.onnx'

/**
 * Determine which player's turn it is for the current game phase.
 */
function activePlayer(state: GameState): number {
  if (state.turnPhase === 'conflictSupport') {
    const conflict = state.pendingConflict!
    return conflict.attackerCommitted === null
      ? conflict.attacker.playerIndex
      : conflict.defender.playerIndex
  }
  return state.currentPlayer
}

/**
 * Run a single game with all AI players to completion. Async because the
 * ONNX path uses session.run() which returns a Promise; the heuristic path
 * is sync but the function is uniformly async to keep one code path.
 */
export async function runGame(playerCountOrOpts: number | RunOptions, maxActions?: number): Promise<GameResult> {
  const opts: RunOptions = typeof playerCountOrOpts === 'number'
    ? { playerCount: playerCountOrOpts, maxActions, log: true }
    : playerCountOrOpts
  const playerCount = opts.playerCount
  const actionLimit = opts.maxActions ?? DEFAULT_MAX_ACTIONS
  const turnLimit = opts.maxTurns ?? Infinity
  const logging = opts.log !== false
  const aiKinds: AiKind[] = opts.aiKinds && opts.aiKinds.length === playerCount
    ? opts.aiKinds
    : new Array(playerCount).fill('simple')
  const onnxModelPath = opts.onnxModelPath ?? DEFAULT_ONNX_MODEL
  const getAction = makeGetAction(aiKinds, onnxModelPath)

  const aiFlags = new Array(playerCount).fill(true)
  let state = createGame(playerCount, aiFlags)
  let actionCount = 0
  let turnNumber = 0
  let lastTurnPlayer = -1
  const log: string[] = []
  const SNAPSHOT_INTERVAL = 10 // score snapshot every N turns

  // Log initial state
  if (logging) {
    log.push(`GAME ${playerCount}p | bag=${state.bag.length}`)
    for (let i = 0; i < playerCount; i++) {
      const p = state.players[i]
      log.push(`  P${i + 1}(${p.dynasty}) hand=${handSummary(p.hand)}`)
    }
  }

  while (state.turnPhase !== 'gameOver' && actionCount < actionLimit) {
    // Detect turn transitions
    const cp = state.currentPlayer
    if (state.turnPhase === 'action' && state.actionsRemaining === 2 && cp !== lastTurnPlayer) {
      if (cp <= lastTurnPlayer || lastTurnPlayer === -1) turnNumber++
      lastTurnPlayer = cp

      if (turnNumber > turnLimit) break

      if (logging) {
        const p = state.players[cp]
        log.push(`=== T${turnNumber} P${cp + 1}(${p.dynasty}) hand=${handSummary(p.hand)} bag=${state.bag.length} ===`)
        if (turnNumber > 0 && turnNumber % SNAPSHOT_INTERVAL === 0) {
          log.push(`  SCORES: ${scoreSnapshot(state.players)}`)
        }
      }
    }

    const playerIndex = activePlayer(state)
    const action: GameAction = await getAction(state, playerIndex)
    const prevState = state
    let fallback = false

    try {
      state = gameReducer(state, { ...action, playerIndex })
    } catch {
      fallback = true
      if (state.turnPhase === 'action') {
        state = gameReducer(state, { type: 'pass', playerIndex })
      } else {
        state = handleFallback(state, playerIndex)
      }
    }

    if (logging) {
      log.push(formatActionLine(action, prevState, state, fallback))
      const cLine = conflictResolutionLine(prevState, state)
      if (cLine) log.push(cLine)
    }

    actionCount++
  }

  // Determine reason
  let reason: GameResult['reason']
  if (state.turnPhase === 'gameOver') reason = 'gameOver'
  else if (turnNumber > turnLimit) reason = 'turnLimit'
  else reason = 'maxTurns'

  // Final score snapshot
  if (logging) {
    log.push(`--- END: ${reason} after ${turnNumber} turns, ${actionCount} actions ---`)
    log.push(`FINAL: ${scoreSnapshot(state.players)}`)
  }

  return buildResult(state, actionCount, turnNumber, reason, log)
}

function handleFallback(state: GameState, playerIndex: number): GameState {
  switch (state.turnPhase) {
    case 'conflictSupport':
      return gameReducer(state, { type: 'commitSupport', indices: [], playerIndex })
    case 'monumentChoice':
      return gameReducer(state, { type: 'declineMonument', playerIndex })
    case 'warOrderChoice': {
      const color = state.pendingConflict?.pendingWarColors?.[0] ?? 'red'
      return gameReducer(state, { type: 'chooseWarOrder', color, playerIndex })
    }
    default:
      return state
  }
}

function buildResult(
  state: GameState,
  actionCount: number,
  turns: number,
  reason: GameResult['reason'],
  log: string[],
): GameResult {
  const scores = state.players.map(p => ({
    dynasty: p.dynasty,
    minScore: Math.min(p.score.red, p.score.blue, p.score.green, p.score.black),
    score: { ...p.score },
    treasures: p.treasures,
  }))

  let winner = 0
  for (let i = 1; i < scores.length; i++) {
    const current = scores[i]
    const best = scores[winner]
    if (
      current.minScore > best.minScore ||
      (current.minScore === best.minScore && current.treasures > best.treasures) ||
      (current.minScore === best.minScore && current.treasures === best.treasures &&
        totalScore(current.score) > totalScore(best.score))
    ) {
      winner = i
    }
  }

  return { winner, scores, turnCount: actionCount, turns, reason, log }
}

function totalScore(score: Record<string, number>): number {
  return Object.values(score).reduce((sum, v) => sum + v, 0)
}

/**
 * Run multiple games and return aggregate results.
 */
export async function runTournament(
  gameCount: number,
  playerCount: number,
  maxActions = DEFAULT_MAX_ACTIONS,
  extra: { aiKinds?: AiKind[]; onnxModelPath?: string } = {},
): Promise<{ results: GameResult[]; wins: number[]; avgMinScores: number[] }> {
  const results: GameResult[] = []
  const wins = new Array(playerCount).fill(0)
  const totalMinScores = new Array(playerCount).fill(0)

  for (let i = 0; i < gameCount; i++) {
    const result = await runGame({
      playerCount, maxActions, log: false,
      aiKinds: extra.aiKinds,
      onnxModelPath: extra.onnxModelPath,
    })
    results.push(result)
    wins[result.winner]++
    for (let p = 0; p < playerCount; p++) {
      totalMinScores[p] += result.scores[p].minScore
    }
  }

  const avgMinScores = totalMinScores.map(t => t / gameCount)

  return { results, wins, avgMinScores }
}
