import { createGame } from '../setup'
import { applyAction } from '../actions'

describe('swapTiles', () => {
  it('removes selected tiles and draws replacements', () => {
    const state = createGame(2)
    const result = applyAction(state, { type: 'swapTiles', indices: [0, 1] })
    expect(result.players[0].hand).toHaveLength(6)
    expect(result.actionsRemaining).toBe(1)
  })

  it('does not mutate original state', () => {
    const state = createGame(2)
    const handBefore = [...state.players[0].hand]
    const actionsBefore = state.actionsRemaining
    applyAction(state, { type: 'swapTiles', indices: [0, 1] })
    expect(state.players[0].hand).toEqual(handBefore)
    expect(state.actionsRemaining).toBe(actionsBefore)
  })

  it('throws if index is out of range', () => {
    const state = createGame(2)
    expect(() => applyAction(state, { type: 'swapTiles', indices: [10] })).toThrow()
  })

  it('throws if no actions remaining', () => {
    const state = createGame(2)
    state.actionsRemaining = 0
    expect(() => applyAction(state, { type: 'swapTiles', indices: [0] })).toThrow()
  })
})

describe('pass', () => {
  it('decrements actions remaining', () => {
    const state = createGame(2)
    const result = applyAction(state, { type: 'pass' })
    expect(result.actionsRemaining).toBe(1)
  })

  it('throws if no actions remaining', () => {
    const state = createGame(2)
    state.actionsRemaining = 0
    expect(() => applyAction(state, { type: 'pass' })).toThrow()
  })
})

describe('withdrawLeader', () => {
  it('removes leader from board and returns to player supply', () => {
    const state = createGame(2)
    state.board[0][9].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders[0].position = { row: 0, col: 9 }
    const result = applyAction(state, { type: 'withdrawLeader', color: 'red' })
    expect(result.board[0][9].leader).toBeNull()
    expect(result.players[0].leaders[0].position).toBeNull()
  })

  it('does not mutate original state', () => {
    const state = createGame(2)
    state.board[0][9].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders[0].position = { row: 0, col: 9 }
    applyAction(state, { type: 'withdrawLeader', color: 'red' })
    expect(state.board[0][9].leader).not.toBeNull()
    expect(state.players[0].leaders[0].position).not.toBeNull()
  })

  it('throws if leader is not on board', () => {
    const state = createGame(2)
    expect(() => applyAction(state, { type: 'withdrawLeader', color: 'red' })).toThrow()
  })

  it('throws if no actions remaining', () => {
    const state = createGame(2)
    state.actionsRemaining = 0
    expect(() => applyAction(state, { type: 'withdrawLeader', color: 'red' })).toThrow()
  })
})
