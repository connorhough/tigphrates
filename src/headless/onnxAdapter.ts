/**
 * Headless ONNX adapter — runs the trained policy via onnxruntime-node so
 * Node-only flows (`npm run headless`) can pit the trained model against
 * simpleAI without spinning up the Python bridge. Shares the encoding +
 * hierarchical-argmax decode with the browser (src/ai/onnxPolicyCore.ts).
 */

import * as ort from 'onnxruntime-node'
import { GameState, GameAction } from '../engine/types'
import { activePlayerIndex } from '../bridge/encoder'
import { buildFeedsForOnnx, pickActionFromOnnxResult } from '../ai/onnxPolicyCore'

let _session: ort.InferenceSession | null = null
let _loadedPath = ''

export async function loadOnnxSession(modelPath: string): Promise<ort.InferenceSession> {
  if (_session && _loadedPath === modelPath) return _session
  _session = await ort.InferenceSession.create(modelPath)
  _loadedPath = modelPath
  return _session
}

export async function getHeadlessOnnxAction(
  state: GameState,
  modelPath: string,
): Promise<GameAction> {
  const session = await loadOnnxSession(modelPath)
  const playerIndex = activePlayerIndex(state)
  const feeds = buildFeedsForOnnx(state, playerIndex, ort.Tensor)
  const result = await session.run(feeds)
  return pickActionFromOnnxResult(state, result)
}
