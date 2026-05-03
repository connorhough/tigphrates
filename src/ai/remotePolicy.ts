import { GameState, GameAction } from '../engine/types'

// Same-host derivation as src/lab/labApi.ts so Tailscale / LAN viewers reach
// the policy server bound to 0.0.0.0:8765 on the host they're already
// connected to (instead of their own loopback).
function defaultUrl(): string {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8765/action'
  const { protocol, hostname } = window.location
  return `${protocol}//${hostname}:8765/action`
}
const DEFAULT_URL = defaultUrl()

/**
 * Calls the local Python policy server (see python/policy_server.py) with
 * the current game state and resolves to the action the trained policy
 * picks. Throws if the server is unreachable; the caller should fall back
 * to the heuristic AI in that case.
 */
export async function getRemoteAIAction(
  state: GameState,
  url: string = DEFAULT_URL,
): Promise<GameAction> {
  const playerIndex = activePlayerForState(state)
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state, playerIndex }),
  })
  if (!res.ok) {
    throw new Error(`Policy server returned ${res.status}`)
  }
  const json = await res.json()
  if (!json.action) {
    throw new Error('Policy server response missing action')
  }
  return json.action as GameAction
}

function activePlayerForState(state: GameState): number {
  if (state.turnPhase === 'conflictSupport' && state.pendingConflict) {
    return state.pendingConflict.attackerCommitted === null
      ? state.pendingConflict.attacker.playerIndex
      : state.pendingConflict.defender.playerIndex
  }
  return state.currentPlayer
}
