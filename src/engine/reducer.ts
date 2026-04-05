import { GameState, GameAction, TurnPhase } from './types'
import { applyAction } from './actions'
import { endTurn } from './turn'

const ACTION_PHASE_ACTIONS = new Set([
  'placeTile', 'placeLeader', 'placeCatastrophe', 'swapTiles', 'pass', 'withdrawLeader',
])

const CONFLICT_SUPPORT_ACTIONS = new Set(['commitSupport'])

const WAR_ORDER_ACTIONS = new Set(['chooseWarOrder'])

const MONUMENT_CHOICE_ACTIONS = new Set(['buildMonument', 'declineMonument'])

const PHASE_ACTIONS: Record<TurnPhase, Set<string>> = {
  action: ACTION_PHASE_ACTIONS,
  conflictSupport: CONFLICT_SUPPORT_ACTIONS,
  warOrderChoice: WAR_ORDER_ACTIONS,
  monumentChoice: MONUMENT_CHOICE_ACTIONS,
  gameOver: new Set(),
}

function validatePlayer(state: GameState, playerIndex: number): void {
  if (state.turnPhase === 'action' || state.turnPhase === 'warOrderChoice' || state.turnPhase === 'monumentChoice') {
    if (playerIndex !== state.currentPlayer) {
      throw new Error(`Not player ${playerIndex}'s turn (current player: ${state.currentPlayer})`)
    }
  } else if (state.turnPhase === 'conflictSupport') {
    const conflict = state.pendingConflict
    if (!conflict) {
      throw new Error('No pending conflict')
    }
    const expectedPlayer = conflict.attackerCommitted === null
      ? conflict.attacker.playerIndex
      : conflict.defender.playerIndex
    if (playerIndex !== expectedPlayer) {
      throw new Error(`Not player ${playerIndex}'s turn to commit support (expected player ${expectedPlayer})`)
    }
  }
}

export function gameReducer(
  state: GameState,
  action: GameAction & { playerIndex: number },
): GameState {
  const { playerIndex, ...gameAction } = action

  // Phase validation
  const allowedActions = PHASE_ACTIONS[state.turnPhase]
  if (!allowedActions || !allowedActions.has(gameAction.type)) {
    throw new Error(`Action '${gameAction.type}' not allowed in '${state.turnPhase}' phase`)
  }

  // Player validation
  validatePlayer(state, playerIndex)

  // Delegate to applyAction (which does structuredClone internally)
  let next = applyAction(state, gameAction as GameAction)

  // Auto end-of-turn
  if (next.turnPhase === 'action' && next.actionsRemaining === 0) {
    next = endTurn(next)
  }

  return next
}
