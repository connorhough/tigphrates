import { GameState, GameAction, TileColor, TurnPhase } from '../engine/types'
import { createGame } from '../engine/setup'
import { gameReducer } from '../engine/reducer'
import { getAIAction } from '../ai/simpleAI'
import { getHeadlessOnnxAction } from './onnxAdapter'

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

// --- Compact log notation ---
// Colors: R=red B=blue G=green K=black
// Actions: T=placeTile L=placeLeader W=withdraw C=catastrophe S=swap P=pass
//          CS=commitSupport WO=warOrder BM=buildMonument DM=declineMonument
// Positions: row.col (e.g. 3.5)
// Score deltas: +1R +2B etc.
// Turn header: === T<n> P<n>(<dynasty>) === (with hand summary)
// Conflicts: REVOLT or WAR with attacker/defender/result
// Fallback actions marked with !fb

const C: Record<string, string> = { red: 'R', blue: 'B', green: 'G', black: 'K' }

function pos(r: number, c: number): string { return `${r}.${c}` }

function handSummary(hand: TileColor[]): string {
  const counts: Record<string, number> = {}
  for (const t of hand) counts[C[t]] = (counts[C[t]] || 0) + 1
  return Object.entries(counts).map(([c, n]) => `${n}${c}`).join('') || '(empty)'
}

function scoreDelta(
  before: Record<TileColor, number>,
  after: Record<TileColor, number>,
): string {
  const parts: string[] = []
  for (const color of ['red', 'blue', 'green', 'black'] as TileColor[]) {
    const diff = after[color] - before[color]
    if (diff > 0) parts.push(`+${diff}${C[color]}`)
  }
  return parts.join(' ')
}

function treasureDelta(before: number, after: number): string {
  const diff = after - before
  return diff > 0 ? ` +${diff}tr` : ''
}

function scoreSnapshot(players: GameState['players']): string {
  return players.map((p, i) =>
    `P${i + 1}[${C.red}${p.score.red} ${C.blue}${p.score.blue} ${C.green}${p.score.green} ${C.black}${p.score.black} tr${p.treasures}]`,
  ).join(' ')
}

function formatAction(action: GameAction): string {
  switch (action.type) {
    case 'placeTile': return `T:${C[action.color]}@${pos(action.position.row, action.position.col)}`
    case 'placeLeader': return `L:${C[action.color]}@${pos(action.position.row, action.position.col)}`
    case 'withdrawLeader': return `W:${C[action.color]}`
    case 'placeCatastrophe': return `C@${pos(action.position.row, action.position.col)}`
    case 'swapTiles': return `S(${action.indices.length})`
    case 'pass': return 'P'
    case 'commitSupport': return `CS(${action.indices.length})`
    case 'chooseWarOrder': return `WO:${C[action.color]}`
    case 'buildMonument': return `BM:${action.monumentId}`
    case 'declineMonument': return 'DM'
    default: return '??'
  }
}

function conflictLine(prev: GameState, next: GameState): string | null {
  // Conflict just resolved: prev had pendingConflict, next doesn't (or has a different one)
  const pc = prev.pendingConflict
  if (!pc) return null
  // Only log when conflict resolves (both sides committed and result applied)
  if (next.pendingConflict === pc) return null
  if (pc.attackerCommitted === null || pc.defenderCommitted === null) return null

  const type = pc.type === 'revolt' ? 'REVOLT' : 'WAR'
  const atkTotal = pc.attackerStrength + pc.attackerCommitted.length
  const defTotal = pc.defenderStrength + pc.defenderCommitted.length
  const winner = atkTotal > defTotal ? 'atk' : 'def'
  return `  ${type}(${C[pc.color]}) P${pc.attacker.playerIndex + 1}(${atkTotal}) vs P${pc.defender.playerIndex + 1}(${defTotal}) -> ${winner} wins`
}

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
      // Log the action
      const actionStr = fallback
        ? `${formatAction(action)}!fb`
        : formatAction(action)

      // Detect score/treasure changes for all players
      const deltas: string[] = []
      for (let i = 0; i < playerCount; i++) {
        const sd = scoreDelta(prevState.players[i].score, state.players[i].score)
        const td = treasureDelta(prevState.players[i].treasures, state.players[i].treasures)
        if (sd || td) deltas.push(`P${i + 1}:${sd}${td}`)
      }
      const deltaStr = deltas.length ? ` [${deltas.join(', ')}]` : ''

      // Phase transition info
      let phaseStr = ''
      if (state.turnPhase !== prevState.turnPhase && state.turnPhase !== 'action') {
        phaseStr = ` ->${state.turnPhase}`
      }

      log.push(`  ${actionStr}${deltaStr}${phaseStr}`)

      // Log conflict resolution
      const cLine = conflictLine(prevState, state)
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
