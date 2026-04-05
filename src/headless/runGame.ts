import { GameState, GameAction } from '../engine/types'
import { createGame } from '../engine/setup'
import { gameReducer } from '../engine/reducer'
import { getAIAction } from '../ai/simpleAI'

export interface GameResult {
  winner: number // player index with highest minimum score
  scores: { dynasty: string; minScore: number; score: Record<string, number>; treasures: number }[]
  turnCount: number
  reason: 'gameOver' | 'maxTurns'
}

const DEFAULT_MAX_ACTIONS = 5000 // safety limit to prevent infinite loops

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
 * Run a single game with all AI players to completion.
 * Returns the game result with scores and winner.
 */
export function runGame(playerCount: number, maxActions = DEFAULT_MAX_ACTIONS): GameResult {
  const aiFlags = new Array(playerCount).fill(true)
  let state = createGame(playerCount, aiFlags)
  let actionCount = 0

  while (state.turnPhase !== 'gameOver' && actionCount < maxActions) {
    const playerIndex = activePlayer(state)
    const action: GameAction = getAIAction(state)

    try {
      state = gameReducer(state, { ...action, playerIndex })
    } catch {
      // If action fails, try passing (only valid in action phase)
      if (state.turnPhase === 'action') {
        state = gameReducer(state, { type: 'pass', playerIndex })
      } else {
        // For non-action phases, try empty commit or decline
        state = handleFallback(state, playerIndex)
      }
    }

    actionCount++
  }

  return buildResult(state, actionCount, state.turnPhase === 'gameOver' ? 'gameOver' : 'maxTurns')
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

function buildResult(state: GameState, actionCount: number, reason: GameResult['reason']): GameResult {
  const scores = state.players.map(p => ({
    dynasty: p.dynasty,
    minScore: Math.min(p.score.red, p.score.blue, p.score.green, p.score.black),
    score: { ...p.score },
    treasures: p.treasures,
  }))

  // Winner is the player with the highest minimum score (ties broken by treasures, then total)
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

  // Approximate turns from action count (rough: 2 actions per turn per player)
  const turnCount = actionCount

  return { winner, scores, turnCount, reason }
}

function totalScore(score: Record<string, number>): number {
  return Object.values(score).reduce((sum, v) => sum + v, 0)
}

/**
 * Run multiple games and return aggregate results.
 */
export function runTournament(
  gameCount: number,
  playerCount: number,
  maxActions = DEFAULT_MAX_ACTIONS,
): { results: GameResult[]; wins: number[]; avgMinScores: number[] } {
  const results: GameResult[] = []
  const wins = new Array(playerCount).fill(0)
  const totalMinScores = new Array(playerCount).fill(0)

  for (let i = 0; i < gameCount; i++) {
    const result = runGame(playerCount, maxActions)
    results.push(result)
    wins[result.winner]++
    for (let p = 0; p < playerCount; p++) {
      totalMinScores[p] += result.scores[p].minScore
    }
  }

  const avgMinScores = totalMinScores.map(t => t / gameCount)

  return { results, wins, avgMinScores }
}
