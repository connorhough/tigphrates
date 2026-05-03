/**
 * Runtime-agnostic ONNX inference core. The browser path uses
 * onnxruntime-web; the headless path uses onnxruntime-node. Both packages
 * re-export `Tensor` and `InferenceSession` from onnxruntime-common, so the
 * tensor-building + decoding logic lives here once and the runtime-specific
 * adapter passes in the session and Tensor class.
 */

import type { InferenceSession, Tensor as OrtTensor } from 'onnxruntime-common'
import { GameState, GameAction } from '../engine/types'
import {
  encodeObservation,
  enumerateValidActions,
  createActionMask,
  ACTION_SPACE_SIZE,
  BOARD_CHANNELS,
  HAND_MAX,
  NUM_ACTION_TYPES,
  TYPE_BASES,
  TYPE_PARAM_SIZES,
} from '../bridge/encoder'

type TensorCtor = typeof OrtTensor

function flattenBoard(board: number[][][]): Float32Array {
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

export function buildFeedsForOnnx(
  state: GameState,
  playerIndex: number,
  TensorClass: TensorCtor,
): Record<string, OrtTensor> {
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
  const oppScoresArr = obs.opponentScores[0] ?? [0, 0, 0, 0]
  const oppLeadersArr = obs.opponentLeaderPositions[0] ?? new Array(8).fill(-1)
  const oppScores = Float32Array.from(oppScoresArr)
  const oppLeaders = Float32Array.from(oppLeadersArr)

  const validActions = enumerateValidActions(state)
  const maskArr = createActionMask(validActions)
  const maskTensor = new Uint8Array(ACTION_SPACE_SIZE)
  for (let i = 0; i < ACTION_SPACE_SIZE; i++) maskTensor[i] = maskArr[i] ? 1 : 0

  return {
    board: new TensorClass('float32', board, [1, BOARD_CHANNELS, obs.board[0].length, obs.board[0][0].length]),
    hand: new TensorClass('float32', hand, [1, 4]),
    hand_seq: new TensorClass('int64', handSeq, [1, HAND_MAX]),
    scores: new TensorClass('float32', scores, [1, 4]),
    meta: new TensorClass('float32', meta, [1, 8]),
    conflict: new TensorClass('float32', conflict, [1, 7]),
    leaders: new TensorClass('float32', leaders, [1, 8]),
    opp_scores: new TensorClass('float32', oppScores, [1, 4]),
    opp_leaders: new TensorClass('float32', oppLeaders, [1, 8]),
    mask: new TensorClass('bool', maskTensor, [1, ACTION_SPACE_SIZE]),
  }
}

/**
 * Decode an ONNX inference result into a GameAction via hierarchical
 * argmax. Mirrors `python/train.py:get_action_and_value`'s argmax path:
 * pick the best valid type from `type_logits`, then the best parameter
 * inside that type's slot range from the (already-mask-applied)
 * `param_logits`.
 */
export function pickActionFromOnnxResult(
  state: GameState,
  result: InferenceSession.OnnxValueMapType,
): GameAction {
  const validActions = enumerateValidActions(state)
  if (validActions.length === 0) {
    throw new Error('No valid actions available')
  }

  const validTypes = new Set<number>()
  for (const a of validActions) validTypes.add(a.typeIdx)

  const typeLogits = (result.type_logits as OrtTensor).data as Float32Array
  let bestType = -1
  let bestTypeVal = -Infinity
  for (let t = 0; t < NUM_ACTION_TYPES; t++) {
    if (!validTypes.has(t)) continue
    if (typeLogits[t] > bestTypeVal) {
      bestTypeVal = typeLogits[t]
      bestType = t
    }
  }
  if (bestType < 0) return validActions[0].action

  const paramLogits = (result.param_logits as OrtTensor).data as Float32Array
  const base = TYPE_BASES[bestType]
  const size = TYPE_PARAM_SIZES[bestType]
  let bestFlat = base
  let bestParamVal = -Infinity
  for (let p = 0; p < size; p++) {
    const v = paramLogits[base + p]
    if (v > bestParamVal) {
      bestParamVal = v
      bestFlat = base + p
    }
  }

  const matched = validActions.find(a => a.index === bestFlat)
  if (!matched) {
    const passAction = validActions.find(a => a.action.type === 'pass')
    if (passAction) return passAction.action
    return validActions[0].action
  }
  return matched.action
}
