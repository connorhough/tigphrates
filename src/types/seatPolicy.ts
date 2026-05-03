/**
 * Per-seat policy configuration. The setup screen and the lab "watch
 * match" form both produce an array of these — one per seat — that App
 * passes through to useGame so the active-player AI can be resolved on
 * each turn.
 */

export type SeatPolicy =
  | { kind: 'human' }
  | { kind: 'heuristic' }
  | { kind: 'server' }                       // python/policy_server.py default model
  | { kind: 'onnx'; modelUrl: string }       // browser ONNX, /policy.onnx by default
                                             // or /pool/<name>.onnx via the lab server

export const DEFAULT_HEURISTIC: SeatPolicy = { kind: 'heuristic' }
export const DEFAULT_HUMAN: SeatPolicy = { kind: 'human' }
