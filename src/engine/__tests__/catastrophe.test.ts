import { createGame } from '../setup'
import { applyAction } from '../actions'
import { GameState } from '../types'

function placeRedTemple(state: GameState, row: number, col: number) {
  state.board[row][col].tile = 'red'
  state.board[row][col].tileFlipped = false
}

describe('catastrophe', () => {
  it('places catastrophe on empty space', () => {
    const state = createGame(2)
    // Pick an empty land cell
    const pos = { row: 5, col: 5 }
    expect(state.board[pos.row][pos.col].tile).toBeNull()

    const result = applyAction(state, { type: 'placeCatastrophe', position: pos })

    expect(result.board[pos.row][pos.col].catastrophe).toBe(true)
    expect(result.board[pos.row][pos.col].tile).toBeNull()
  })

  it('destroys face-up tile when placed on one', () => {
    const state = createGame(2)
    const pos = { row: 5, col: 5 }
    state.board[pos.row][pos.col].tile = 'green'

    const result = applyAction(state, { type: 'placeCatastrophe', position: pos })

    expect(result.board[pos.row][pos.col].catastrophe).toBe(true)
    expect(result.board[pos.row][pos.col].tile).toBeNull()
  })

  it('cannot place on leader', () => {
    const state = createGame(2)
    const pos = { row: 5, col: 5 }
    state.board[pos.row][pos.col].leader = { color: 'red', dynasty: 'archer' }

    expect(() =>
      applyAction(state, { type: 'placeCatastrophe', position: pos })
    ).toThrow()
  })

  it('cannot place on treasure', () => {
    const state = createGame(2)
    // Starting temples have hasTreasure = true, e.g. (0,10)
    const pos = { row: 0, col: 10 }
    expect(state.board[pos.row][pos.col].hasTreasure).toBe(true)

    expect(() =>
      applyAction(state, { type: 'placeCatastrophe', position: pos })
    ).toThrow()
  })

  it('cannot place on monument', () => {
    const state = createGame(2)
    const pos = { row: 5, col: 5 }
    state.board[pos.row][pos.col].monument = 'red-blue'

    expect(() =>
      applyAction(state, { type: 'placeCatastrophe', position: pos })
    ).toThrow()
  })

  it('cannot place on existing catastrophe', () => {
    const state = createGame(2)
    const pos = { row: 5, col: 5 }
    state.board[pos.row][pos.col].catastrophe = true

    expect(() =>
      applyAction(state, { type: 'placeCatastrophe', position: pos })
    ).toThrow()
  })

  it('splits kingdom when connectivity broken', () => {
    const state = createGame(2)
    // Build a line of tiles: (5,3), (5,4), (5,5)
    // Place a leader at (5,2) adjacent to (5,3) which has a red temple
    placeRedTemple(state, 5, 3)
    state.board[5][4].tile = 'green'
    placeRedTemple(state, 5, 5)

    // Place leaders on each end
    state.board[5][2].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders[0].position = { row: 5, col: 2 }
    state.board[5][6].leader = { color: 'blue', dynasty: 'bull' }
    state.players[1].leaders[1].position = { row: 5, col: 6 }

    // Catastrophe on the middle tile should split the kingdom
    const result = applyAction(state, { type: 'placeCatastrophe', position: { row: 5, col: 4 } })

    expect(result.board[5][4].catastrophe).toBe(true)
    expect(result.board[5][4].tile).toBeNull()
  })

  it('leader withdrawn if stranded by catastrophe', () => {
    const state = createGame(2)
    // Place a single red temple at (5,5), leader at (5,4) adjacent to it
    placeRedTemple(state, 5, 5)
    state.board[5][4].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders[0].position = { row: 5, col: 4 }

    // Catastrophe on the red temple strands the leader
    const result = applyAction(state, { type: 'placeCatastrophe', position: { row: 5, col: 5 } })

    expect(result.board[5][4].leader).toBeNull()
    expect(result.players[0].leaders[0].position).toBeNull()
  })

  it('decrements player catastrophe count', () => {
    const state = createGame(2)
    expect(state.players[0].catastrophesRemaining).toBe(2)

    const result = applyAction(state, { type: 'placeCatastrophe', position: { row: 5, col: 5 } })

    expect(result.players[0].catastrophesRemaining).toBe(1)
  })

  it('rejects when no catastrophes remaining', () => {
    const state = createGame(2)
    state.players[0].catastrophesRemaining = 0

    expect(() =>
      applyAction(state, { type: 'placeCatastrophe', position: { row: 5, col: 5 } })
    ).toThrow()
  })

  it('uses 1 action', () => {
    const state = createGame(2)
    expect(state.actionsRemaining).toBe(2)

    const result = applyAction(state, { type: 'placeCatastrophe', position: { row: 5, col: 5 } })

    expect(result.actionsRemaining).toBe(1)
  })

  it('throws if no actions remaining', () => {
    const state = createGame(2)
    state.actionsRemaining = 0

    expect(() =>
      applyAction(state, { type: 'placeCatastrophe', position: { row: 5, col: 5 } })
    ).toThrow('No actions remaining')
  })
})
