/**
 * In-browser trained-policy adapter.
 *
 * Loads the ONNX-exported PolicyValueNetwork via onnxruntime-web, encodes
 * the current game state via the shared bridge encoder, runs inference, and
 * decodes the chosen action index back to a dispatch-ready GameAction.
 *
 * No Python server required; matches the encoding used during training so
 * the same model file works in both contexts.
 */

import * as ort from 'onnxruntime-web'
import { GameState, GameAction } from '../engine/types'
import {
  encodeObservation,
  enumerateValidActions,
  createActionMask,
  activePlayerIndex,
  ACTION_SPACE_SIZE,
  BOARD_CHANNELS,
  HAND_MAX,
} from '../bridge/encoder'

const DEFAULT_MODEL_URL = '/policy.onnx'

let _sessionPromise: Promise<ort.InferenceSession> | null = null

function getSession(modelUrl: string = DEFAULT_MODEL_URL): Promise<ort.InferenceSession> {
  if (_sessionPromise === null) {
    _sessionPromise = ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  }
  return _sessionPromise
}

function flattenBoard(board: number[][][]): Float32Array {
  // (C, H, W) → flat (C*H*W). Shape ordering matches what export_onnx.py
  // expects from the FlatPolicy wrapper (PyTorch's default channels-first).
  const C = board.length
  const H = board[0].length
  const W = board[0][0].length
  const out = new Float32Array(C * H * W)
  let idx = 0
  for (let c = 0; c < C; c++) {
    for (let r = 0; r < H; r++) {
      for (let cc = 0; cc < W; cc++) {
        out[idx++] = board[c][r][cc]
      }
    }
  }
  return out
}

function buildFeeds(state: GameState, playerIndex: number): Record<string, ort.Tensor> {
  const obs = encodeObservation(state, playerIndex)

  const board = flattenBoard(obs.board)
  const hand = Float32Array.from(obs.hand)
  const handSeq = BigInt64Array.from(obs.handSeq.map(v => BigInt(v)))
  const scores = Float32Array.from(obs.scores)
  const meta = Float32Array.from([
    obs.treasures, obs.catastrophesRemaining, obs.bagSize,
    obs.actionsRemaining, obs.turnPhase, obs.currentPlayer,
    obs.playerIndex, obs.numPlayers,
  ])
  const conflict = obs.conflict
    ? Float32Array.from([
        obs.conflict.type, obs.conflict.color,
        obs.conflict.attackerStrength, obs.conflict.defenderStrength,
        obs.conflict.attackerCommitted ? 1 : 0,
        obs.conflict.isAttacker ? 1 : 0,
        obs.conflict.isDefender ? 1 : 0,
      ])
    : new Float32Array(7)
  const leaders = Float32Array.from(obs.leaderPositions)
  // Match the env: opp_scores / opp_leaders are the FIRST opponent only
  // (2-player training shape). Build with zero / -1 padding for empty slots.
  const oppScoresArr = obs.opponentScores[0] ?? [0, 0, 0, 0]
  const oppLeadersArr = obs.opponentLeaderPositions[0] ?? new Array(8).fill(-1)
  const oppScores = Float32Array.from(oppScoresArr)
  const oppLeaders = Float32Array.from(oppLeadersArr)

  const validActions = enumerateValidActions(state)
  const maskArr = createActionMask(validActions)
  const maskTensor = new Uint8Array(ACTION_SPACE_SIZE)
  for (let i = 0; i < ACTION_SPACE_SIZE; i++) maskTensor[i] = maskArr[i] ? 1 : 0

  return {
    board: new ort.Tensor('float32', board, [1, BOARD_CHANNELS, obs.board[0].length, obs.board[0][0].length]),
    hand: new ort.Tensor('float32', hand, [1, 4]),
    hand_seq: new ort.Tensor('int64', handSeq, [1, HAND_MAX]),
    scores: new ort.Tensor('float32', scores, [1, 4]),
    meta: new ort.Tensor('float32', meta, [1, 8]),
    conflict: new ort.Tensor('float32', conflict, [1, 7]),
    leaders: new ort.Tensor('float32', leaders, [1, 8]),
    opp_scores: new ort.Tensor('float32', oppScores, [1, 4]),
    opp_leaders: new ort.Tensor('float32', oppLeaders, [1, 8]),
    mask: new ort.Tensor('bool', maskTensor, [1, ACTION_SPACE_SIZE]),
  }
}

/**
 * Calls the ONNX-exported policy in-browser to pick an action for the
 * current active player. Throws if the model can't load (caller should fall
 * back to the heuristic).
 */
export async function getOnnxAIAction(
  state: GameState,
  modelUrl: string = DEFAULT_MODEL_URL,
): Promise<GameAction> {
  const playerIndex = activePlayerIndex(state)
  const session = await getSession(modelUrl)
  const feeds = buildFeeds(state, playerIndex)
  const result = await session.run(feeds)

  const logits = result.logits.data as Float32Array
  // Argmax — logits are pre-masked in the export graph (-1e9 for invalid).
  let bestIdx = 0
  let bestVal = -Infinity
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i]
      bestIdx = i
    }
  }

  // Decode index → GameAction via the same enumerator used in the bridge.
  const validActions = enumerateValidActions(state)
  const matched = validActions.find(a => a.index === bestIdx)
  if (!matched) {
    // Fallback: model picked an invalid index (shouldn't happen with masking
    // but be defensive). Pick any valid action — prefer 'pass' if present.
    const passAction = validActions.find(a => a.action.type === 'pass')
    if (passAction) return passAction.action
    if (validActions.length > 0) return validActions[0].action
    throw new Error('No valid actions available')
  }
  return matched.action
}
