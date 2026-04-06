import { createGame } from '../setup'
import { gameReducer } from '../reducer'

describe('gameReducer', () => {
  it('rejects actions from wrong player', () => {
    const state = createGame(2)
    // Player 1 tries to act on player 0's turn
    expect(() =>
      gameReducer(state, { type: 'pass', playerIndex: 1 })
    ).toThrow()
  })

  it('rejects actions in wrong phase', () => {
    const state = createGame(2)
    // Force conflict support phase
    state.turnPhase = 'conflictSupport'
    state.pendingConflict = {
      type: 'revolt',
      color: 'red',
      attacker: { playerIndex: 0, position: { row: 0, col: 9 } },
      defender: { playerIndex: 1, position: { row: 1, col: 2 } },
      attackerStrength: 1,
      defenderStrength: 1,
      attackerCommitted: null,
      defenderCommitted: null,
    }
    // Try to place tile during conflictSupport phase
    expect(() =>
      gameReducer(state, { type: 'pass', playerIndex: 0 })
    ).toThrow()
  })

  it('auto-calls endTurn when actions exhausted', () => {
    const state = createGame(2)
    state.currentPlayer = 0
    state.actionsRemaining = 2

    // Pass once
    let next = gameReducer(state, { type: 'pass', playerIndex: 0 })
    expect(next.actionsRemaining).toBe(1)
    expect(next.currentPlayer).toBe(0)

    // Pass again — should auto end turn
    next = gameReducer(next, { type: 'pass', playerIndex: 0 })
    // Turn should have advanced to player 1
    expect(next.currentPlayer).toBe(1)
    expect(next.actionsRemaining).toBe(2)
    // Player 0's hand should be refilled to 6
    expect(state.players[0].hand).toHaveLength(6) // original unchanged
    expect(next.players[0].hand).toHaveLength(6)
  })

  it('full turn: place tile, place leader, draw, next player', () => {
    const state = createGame(2)
    // Give player 0 a known hand with a non-blue tile for land placement
    state.players[0].hand = ['black', 'green', 'red', 'blue', 'black', 'green']

    // Place a black tile on an empty land cell
    let next = gameReducer(state, {
      type: 'placeTile',
      color: 'black',
      position: { row: 5, col: 5 },
      playerIndex: 0,
    })
    expect(next.actionsRemaining).toBe(1)
    expect(next.players[0].hand).toHaveLength(5)

    // Pass second action
    next = gameReducer(next, { type: 'pass', playerIndex: 0 })
    // Turn should advance to player 1
    expect(next.currentPlayer).toBe(1)
    expect(next.actionsRemaining).toBe(2)
    // Player 0 hand refilled to 6
    expect(next.players[0].hand).toHaveLength(6)
  })

  it('full turn with revolt: place leader triggers conflict, resolve, continue', () => {
    const state = createGame(2)

    // Setup: place a red temple tile at (1,2) which is land, adjacent to existing temple at (1,1)
    // Player 1's red leader is at (0,9) adjacent to temple at (0,10)
    state.board[0][9].leader = { color: 'red', dynasty: 'bull' }
    state.players[1].leaders[0].position = { row: 0, col: 9 }

    // Player 0 places their red leader adjacent to a temple near player 1's red leader
    // We need them in the same kingdom. Put player 0's red leader adjacent to temple at (0,10)
    // and ensure they share a kingdom with player 1's red leader at (0,9)
    // (0,10) has a temple tile; (0,9) has player 1's red leader; (0,11) is empty land
    // Place player 0's red leader at (0,11) — adjacent to temple at (0,10)
    state.players[0].leaders[0].position = null // ensure red leader is off board

    // Give player 0 some red tiles for support
    state.players[0].hand = ['red', 'red', 'red', 'blue', 'black', 'green']
    state.players[1].hand = ['red', 'red', 'blue', 'blue', 'black', 'green']

    let next = gameReducer(state, {
      type: 'placeLeader',
      color: 'red',
      position: { row: 0, col: 11 },
      playerIndex: 0,
    })

    // Should be in conflict support phase (revolt)
    expect(next.turnPhase).toBe('conflictSupport')
    expect(next.pendingConflict).not.toBeNull()
    expect(next.pendingConflict!.type).toBe('revolt')

    // Attacker (player 0) commits support
    next = gameReducer(next, {
      type: 'commitSupport',
      indices: [0, 1], // commit 2 red tiles
      playerIndex: 0,
    })
    expect(next.turnPhase).toBe('conflictSupport')
    expect(next.pendingConflict!.attackerCommitted).toEqual([0, 1])

    // Defender (player 1) commits support
    next = gameReducer(next, {
      type: 'commitSupport',
      indices: [0], // commit 1 red tile
      playerIndex: 1,
    })

    // Conflict should be resolved, back to action phase
    expect(next.turnPhase).toBe('action')
  })

  it('allows commit support from correct player during conflict', () => {
    const state = createGame(2)
    state.turnPhase = 'conflictSupport'
    state.pendingConflict = {
      type: 'revolt',
      color: 'red',
      attacker: { playerIndex: 0, position: { row: 0, col: 11 } },
      defender: { playerIndex: 1, position: { row: 0, col: 9 } },
      attackerStrength: 1,
      defenderStrength: 1,
      attackerCommitted: null,
      defenderCommitted: null,
    }
    // Set up board positions
    state.board[0][11].leader = { color: 'red', dynasty: 'archer' }
    state.board[0][9].leader = { color: 'red', dynasty: 'bull' }
    state.players[0].leaders[0].position = { row: 0, col: 11 }
    state.players[1].leaders[0].position = { row: 0, col: 9 }
    state.players[0].hand = ['red', 'red', 'blue', 'blue', 'black', 'green']
    state.players[1].hand = ['red', 'blue', 'blue', 'blue', 'black', 'green']

    // Attacker (player 0) commits first — should succeed
    const next = gameReducer(state, {
      type: 'commitSupport',
      indices: [0],
      playerIndex: 0,
    })
    expect(next.pendingConflict!.attackerCommitted).toEqual([0])
  })

  it('rejects commit support from wrong player during conflict', () => {
    const state = createGame(2)
    state.turnPhase = 'conflictSupport'
    state.pendingConflict = {
      type: 'revolt',
      color: 'red',
      attacker: { playerIndex: 0, position: { row: 0, col: 11 } },
      defender: { playerIndex: 1, position: { row: 0, col: 9 } },
      attackerStrength: 1,
      defenderStrength: 1,
      attackerCommitted: null,
      defenderCommitted: null,
    }

    // Defender (player 1) tries to commit before attacker — should fail
    expect(() =>
      gameReducer(state, {
        type: 'commitSupport',
        indices: [0],
        playerIndex: 1,
      })
    ).toThrow()
  })

  it('rejects all actions in gameOver phase', () => {
    const state = createGame(2)
    state.turnPhase = 'gameOver'
    expect(() =>
      gameReducer(state, { type: 'pass', playerIndex: 0 })
    ).toThrow()
  })

  it('does not double-clone — mutations from applyAction are preserved', () => {
    const state = createGame(2)
    const result = gameReducer(state, { type: 'pass', playerIndex: 0 })
    // Original state should not be mutated (applyAction clones)
    expect(state.actionsRemaining).toBe(2)
    expect(result.actionsRemaining).toBe(1)
  })
})
