import { createGame } from '../setup'
import { applyAction } from '../actions'
import { GameState } from '../types'

/**
 * Helper: create a 2-player game with a deterministic setup for revolt tests.
 * Places a red temple tile adjacent to a target position so leaders can be placed there.
 */
function setupForLeaderPlacement(): GameState {
  const state = createGame(2)
  // Starting temples exist at known positions (see types.ts STARTING_TEMPLES).
  // (0,10) is a starting temple (red tile). Adjacent land cells: (0,9), (0,11), (1,10).
  // (1,1) is a starting temple. Adjacent land cells: (0,1), (1,0), (1,2), (2,1).
  return state
}

describe('placeLeader', () => {
  it('places leader on empty land adjacent to temple', () => {
    const state = setupForLeaderPlacement()
    // (0,9) is land, adjacent to temple at (0,10)
    const result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 0, col: 9 } })
    expect(result.board[0][9].leader).toEqual({ color: 'red', dynasty: 'archer' })
    expect(result.players[0].leaders.find(l => l.color === 'red')!.position).toEqual({ row: 0, col: 9 })
    expect(result.actionsRemaining).toBe(1)
  })

  it('rejects placement not adjacent to face-up red temple', () => {
    const state = setupForLeaderPlacement()
    // (5,5) is land but not adjacent to any temple
    expect(() => applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 5, col: 5 } })).toThrow(
      /adjacent.*temple/i
    )
  })

  it('rejects placement on non-land cell', () => {
    const state = setupForLeaderPlacement()
    // (0,4) is a river cell
    expect(() => applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 0, col: 4 } })).toThrow()
  })

  it('rejects placement on occupied cell', () => {
    const state = setupForLeaderPlacement()
    // (0,10) has a temple tile
    expect(() => applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 0, col: 10 } })).toThrow()
  })

  it('repositions an on-board leader to a new valid cell', () => {
    const state = setupForLeaderPlacement()
    // Place red leader first at (0,9), adjacent to temple (0,10)
    state.board[0][9].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 0, col: 9 }
    // Reposition to (0,11), also adjacent to temple (0,10)
    const result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 0, col: 11 } })
    expect(result.board[0][9].leader).toBeNull()
    expect(result.board[0][11].leader).toEqual({ color: 'red', dynasty: 'archer' })
    expect(result.players[0].leaders.find(l => l.color === 'red')!.position).toEqual({ row: 0, col: 11 })
  })

  it('rejects repositioning a leader to its own current position', () => {
    const state = setupForLeaderPlacement()
    state.board[0][9].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 0, col: 9 }
    expect(() => applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 0, col: 9 } })).toThrow()
  })

  it('rejects placement that would unite two kingdoms', () => {
    const state = setupForLeaderPlacement()
    // Create two separate kingdoms with leaders.
    // Kingdom 1: around temple at (1,1). Place player 1 (bull) green leader at (0,1).
    state.board[0][1].leader = { color: 'green', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'green')!.position = { row: 0, col: 1 }

    // Kingdom 2: place a tile at (2,2) to extend from temple at (2,5) — actually let's be more direct.
    // Place a tile at (1,3) and a leader nearby to form a second kingdom.
    state.board[1][3].tile = 'green'
    state.board[0][3].leader = { color: 'blue', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'blue')!.position = { row: 0, col: 3 }

    // Now (1,2) is adjacent to both kingdoms (kingdom at (0,1)+(1,1) and kingdom at (0,3)+(1,3)).
    // We need a temple adjacent to (1,2) — (1,1) is a temple, so that's satisfied.
    // Placing a leader at (1,2) would connect both kingdoms.
    expect(() => applyAction(state, { type: 'placeLeader', color: 'black', position: { row: 1, col: 2 } })).toThrow(
      /unite|unify|two.*kingdom/i
    )
  })

  it('does not mutate original state', () => {
    const state = setupForLeaderPlacement()
    const boardBefore = state.board[0][9].leader
    applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 0, col: 9 } })
    expect(state.board[0][9].leader).toBe(boardBefore)
  })

  it('repositioning into kingdom with same-color enemy leader triggers revolt', () => {
    const state = setupForLeaderPlacement()
    // Player 1 (bull) places a red leader at (0,1) adjacent to temple (1,1)
    state.board[0][1].leader = { color: 'red', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'red')!.position = { row: 0, col: 1 }
    // Player 0 (archer) has red leader off-board; place at (1,0) — same kingdom (both adjacent to temple (1,1)),
    // triggering a revolt.
    state.currentPlayer = 0
    state.actionsRemaining = 2
    const result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 1, col: 0 } })
    expect(result.turnPhase).toBe('conflictSupport')
    expect(result.pendingConflict?.type).toBe('revolt')
    expect(result.pendingConflict?.color).toBe('red')
  })

  it('throws if no actions remaining', () => {
    const state = setupForLeaderPlacement()
    state.actionsRemaining = 0
    expect(() => applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 0, col: 9 } })).toThrow()
  })
})

describe('revolt', () => {
  function setupRevolt(): GameState {
    const state = createGame(2)
    // Temple at (1,1). Place player 1's red leader at (0,1) (adjacent to temple).
    state.board[0][1].leader = { color: 'red', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'red')!.position = { row: 0, col: 1 }
    // currentPlayer is 0 (archer). They will place their red leader adjacent to same temple → revolt.
    // (2,1) is adjacent to temple at (1,1) and is land.
    return state
  }

  it('triggers revolt when placing leader in kingdom with same-colored leader', () => {
    const state = setupRevolt()
    const result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 2, col: 1 } })
    expect(result.turnPhase).toBe('conflictSupport')
    expect(result.pendingConflict).not.toBeNull()
    expect(result.pendingConflict!.type).toBe('revolt')
    expect(result.pendingConflict!.color).toBe('red')
    expect(result.pendingConflict!.attacker.playerIndex).toBe(0)
    expect(result.pendingConflict!.defender.playerIndex).toBe(1)
    // actionsRemaining should NOT be decremented yet
    expect(result.actionsRemaining).toBe(2)
  })

  it('attacker wins with more adjacent temples + support', () => {
    const state = setupRevolt()
    // Add extra temple adjacent to attacker position (2,1) to give attacker base strength advantage.
    // Place a red temple at (2,0) — adjacent to attacker at (2,1).
    state.board[2][0].tile = 'red'

    // Give player 0 some red tiles in hand for support
    state.players[0].hand = ['red', 'red', 'red', 'blue', 'green', 'black']
    state.players[1].hand = ['blue', 'blue', 'green', 'green', 'black', 'black']

    // Place leader → revolt
    let result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 2, col: 1 } })
    expect(result.turnPhase).toBe('conflictSupport')

    // Attacker base strength: adjacent temples to (2,1) → (1,1) is temple, (2,0) is temple = 2
    // Defender base strength: adjacent temples to (0,1) → (1,1) is temple = 1
    expect(result.pendingConflict!.attackerStrength).toBe(2)
    expect(result.pendingConflict!.defenderStrength).toBe(1)

    // Attacker commits 1 red tile (indices [0])
    result = applyAction(result, { type: 'commitSupport', indices: [0] })
    expect(result.pendingConflict!.attackerCommitted).toEqual([0])
    expect(result.turnPhase).toBe('conflictSupport')

    // Defender commits 0 red tiles
    result = applyAction(result, { type: 'commitSupport', indices: [] })

    // Revolt resolved: attacker wins (2+1=3 vs 1+0=1)
    expect(result.turnPhase).toBe('action')
    expect(result.pendingConflict).toBeNull()
    // Defender's red leader should be removed
    expect(result.players[1].leaders.find(l => l.color === 'red')!.position).toBeNull()
    expect(result.board[0][1].leader).toBeNull()
    // Attacker's red leader should remain
    expect(result.board[2][1].leader).toEqual({ color: 'red', dynasty: 'archer' })
    // Winner gets 1 red VP
    expect(result.players[0].score.red).toBe(1)
    // actionsRemaining decremented after resolution
    expect(result.actionsRemaining).toBe(1)
    // Attacker's committed red tile removed from hand
    expect(result.players[0].hand).toEqual(['red', 'red', 'blue', 'green', 'black'])
  })

  it('defender wins ties', () => {
    const state = setupRevolt()
    // Both have 1 adjacent temple (at (1,1)). No support committed → tie → defender wins.
    state.players[0].hand = ['blue', 'blue', 'green', 'green', 'black', 'black']
    state.players[1].hand = ['blue', 'blue', 'green', 'green', 'black', 'black']

    let result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 2, col: 1 } })
    // Both base strength = 1
    expect(result.pendingConflict!.attackerStrength).toBe(1)
    expect(result.pendingConflict!.defenderStrength).toBe(1)

    // Attacker commits 0
    result = applyAction(result, { type: 'commitSupport', indices: [] })
    // Defender commits 0
    result = applyAction(result, { type: 'commitSupport', indices: [] })

    // Defender wins (tie goes to defender)
    expect(result.turnPhase).toBe('action')
    expect(result.pendingConflict).toBeNull()
    // Attacker's leader removed
    expect(result.players[0].leaders.find(l => l.color === 'red')!.position).toBeNull()
    expect(result.board[2][1].leader).toBeNull()
    // Defender's leader stays
    expect(result.board[0][1].leader).toEqual({ color: 'red', dynasty: 'bull' })
    // Defender gets 1 red VP
    expect(result.players[1].score.red).toBe(1)
    expect(result.actionsRemaining).toBe(1)
  })

  it('only counts face-up red temples as base strength', () => {
    const state = setupRevolt()
    // Flip the temple at (1,1) so it shouldn't count
    state.board[1][1].tileFlipped = true

    // Add a face-up red temple at (2,0) adjacent to attacker
    state.board[2][0].tile = 'red'

    let result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 2, col: 1 } })
    // Attacker: (2,0) is face-up red temple, (1,1) is flipped → only 1
    // Defender: (1,1) is flipped → 0
    expect(result.pendingConflict!.attackerStrength).toBe(1)
    expect(result.pendingConflict!.defenderStrength).toBe(0)
  })

  it('committed tiles removed from game (not returned to bag)', () => {
    const state = setupRevolt()
    state.players[0].hand = ['red', 'red', 'blue', 'green', 'black', 'black']
    state.players[1].hand = ['red', 'blue', 'green', 'green', 'black', 'black']
    const bagSizeBefore = state.bag.length

    let result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 2, col: 1 } })
    // Attacker commits 2 red tiles
    result = applyAction(result, { type: 'commitSupport', indices: [0, 1] })
    // Defender commits 1 red tile
    result = applyAction(result, { type: 'commitSupport', indices: [0] })

    // 3 total red tiles committed — should not be in bag
    expect(result.bag.length).toBe(bagSizeBefore)
    // Attacker's hand lost 2 red tiles
    expect(result.players[0].hand.filter(t => t === 'red').length).toBe(0)
    // Defender's hand lost 1 red tile
    expect(result.players[1].hand.filter(t => t === 'red').length).toBe(0)
  })

  it('rejects commitSupport with non-red tiles during revolt', () => {
    const state = setupRevolt()
    state.players[0].hand = ['red', 'blue', 'green', 'green', 'black', 'black']

    let result = applyAction(state, { type: 'placeLeader', color: 'red', position: { row: 2, col: 1 } })
    // Try to commit a blue tile (index 1)
    expect(() => applyAction(result, { type: 'commitSupport', indices: [1] })).toThrow(/red/i)
  })

  it('rejects commitSupport when not in conflictSupport phase', () => {
    const state = setupForLeaderPlacement()
    expect(() => applyAction(state, { type: 'commitSupport', indices: [] })).toThrow()
  })
})
