/**
 * In-browser trained-policy adapter.
 *
 * Loads ONNX-exported PolicyValueNetwork files via onnxruntime-web. Each
 * model URL gets its own cached `InferenceSession`, so the Lab "watch
 * model A vs model B" flow can keep two policies hot without reloading.
 * Encoding + hierarchical-argmax decode live in onnxPolicyCore.ts.
 */

import * as ort from 'onnxruntime-web'
import { GameState, GameAction } from '../engine/types'
import { activePlayerIndex } from '../bridge/encoder'
import { buildFeedsForOnnx, pickActionFromOnnxResult } from './onnxPolicyCore'

const DEFAULT_MODEL_URL = '/policy.onnx'

const _sessions: Map<string, Promise<ort.InferenceSession>> = new Map()

function getSession(modelUrl: string): Promise<ort.InferenceSession> {
  let p = _sessions.get(modelUrl)
  if (!p) {
    p = ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
    _sessions.set(modelUrl, p)
  }
  return p
}

export async function getOnnxAIAction(
  state: GameState,
  modelUrl: string = DEFAULT_MODEL_URL,
): Promise<GameAction> {
  const playerIndex = activePlayerIndex(state)
  const session = await getSession(modelUrl)
  const feeds = buildFeedsForOnnx(state, playerIndex, ort.Tensor)
  const result = await session.run(feeds)
  return pickActionFromOnnxResult(state, result)
}
