import { createGame } from '../setup'
import { applyAction } from '../actions'
import { GameState } from '../types'

function makeTestState(): GameState {
  const state = createGame(2)
  // Give player 0 known tiles
  state.players[0].hand = ['black', 'green', 'red', 'blue', 'black', 'green']
  return state
}

describe('placeTile', () => {
  it('places tile on empty land space', () => {
    const state = makeTestState()
    // (0,0) is land, empty
    const result = applyAction(state, { type: 'placeTile', color: 'black', position: { row: 0, col: 0 } })
    expect(result.board[0][0].tile).toBe('black')
  })

  it('scores 1 VP when matching leader in kingdom', () => {
    const state = makeTestState()
    // Place a red temple at (1,0) already exists? No. Let's set up:
    // Put a red tile at (1,0) and a black leader (king) at (1,1) adjacent to it (leaders need adjacent temple)
    // Actually let's set up a kingdom with a black leader and place a black tile into it
    // Place red tile at (9,0) as a temple, black leader at (9,1) adjacent to it
    state.board[9][0].tile = 'red'
    state.board[9][1].leader = { color: 'black', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'black')!.position = { row: 9, col: 1 }

    // Now place a black tile at (9,2) — adjacent to the leader, joins the kingdom
    // (9,2) should be land and empty
    expect(state.board[9][2].terrain).toBe('land')
    expect(state.board[9][2].tile).toBeNull()

    const result = applyAction(state, { type: 'placeTile', color: 'black', position: { row: 9, col: 2 } })
    // Black leader's owner (player 0) should get +1 black VP
    expect(result.players[0].score.black).toBe(1)
  })

  it('king scores 1 black VP when no matching leader', () => {
    const state = makeTestState()
    // Set up kingdom with only black leader (king), place green tile in it
    state.board[9][0].tile = 'red'
    state.board[9][1].leader = { color: 'black', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'black')!.position = { row: 9, col: 1 }

    // Place green tile adjacent to the kingdom
    const result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 2 } })
    // King (black leader) collects +1 black VP for unmatched color
    expect(result.players[0].score.black).toBe(1)
  })

  it('no VP when no leader in kingdom', () => {
    const state = makeTestState()
    // Place a black tile at (9,0) — no leaders nearby, just a region
    state.board[9][0].tile = 'red' // existing tile

    const scoreBefore = { ...state.players[0].score }
    const result = applyAction(state, { type: 'placeTile', color: 'black', position: { row: 9, col: 1 } })
    // No leaders in this region, so no VP
    expect(result.players[0].score).toEqual(scoreBefore)
  })

  it('rejects blue tile on land', () => {
    const state = makeTestState()
    // (0,0) is land
    expect(() =>
      applyAction(state, { type: 'placeTile', color: 'blue', position: { row: 0, col: 0 } })
    ).toThrow()
  })

  it('rejects non-blue tile on river', () => {
    const state = makeTestState()
    // (0,4) is river
    expect(state.board[0][4].terrain).toBe('river')
    expect(() =>
      applyAction(state, { type: 'placeTile', color: 'black', position: { row: 0, col: 4 } })
    ).toThrow()
  })

  it('removes tile from player hand', () => {
    const state = makeTestState()
    expect(state.players[0].hand.filter(t => t === 'black').length).toBe(2)
    const result = applyAction(state, { type: 'placeTile', color: 'black', position: { row: 0, col: 0 } })
    expect(result.players[0].hand.filter(t => t === 'black').length).toBe(1)
  })

  it('decrements actionsRemaining', () => {
    const state = makeTestState()
    const result = applyAction(state, { type: 'placeTile', color: 'black', position: { row: 0, col: 0 } })
    expect(result.actionsRemaining).toBe(1)
  })

  it('throws if cell is occupied', () => {
    const state = makeTestState()
    state.board[0][0].tile = 'red'
    expect(() =>
      applyAction(state, { type: 'placeTile', color: 'black', position: { row: 0, col: 0 } })
    ).toThrow()
  })

  it('throws if player does not have tile in hand', () => {
    const state = makeTestState()
    state.players[0].hand = ['red', 'red', 'red']
    expect(() =>
      applyAction(state, { type: 'placeTile', color: 'black', position: { row: 0, col: 0 } })
    ).toThrow()
  })

  it('throws if no actions remaining', () => {
    const state = makeTestState()
    state.actionsRemaining = 0
    expect(() =>
      applyAction(state, { type: 'placeTile', color: 'black', position: { row: 0, col: 0 } })
    ).toThrow()
  })

  it('throws if tile would unite 3 or more kingdoms', () => {
    const state = makeTestState()
    state.players[0].hand = ['red', 'red', 'red', 'red', 'red', 'red']

    // Three separate kingdoms, each with a red temple + adjacent leader,
    // all flanking the empty land cell (5,5) without touching each other.
    state.board[3][5].tile = 'red'
    state.board[4][5].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 4, col: 5 }

    state.board[5][3].tile = 'red'
    state.board[5][4].leader = { color: 'green', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'green')!.position = { row: 5, col: 4 }

    state.board[5][7].tile = 'red'
    state.board[5][6].leader = { color: 'blue', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'blue')!.position = { row: 5, col: 6 }

    expect(state.board[5][5].terrain).toBe('land')
    expect(state.board[5][5].tile).toBeNull()

    expect(() =>
      applyAction(state, { type: 'placeTile', color: 'red', position: { row: 5, col: 5 } })
    ).toThrow(/3 or more kingdoms|unite/i)
  })

  it('matching color leader scores over king', () => {
    const state = makeTestState()
    // Kingdom with both green leader and black leader (king)
    // Place green tile — green leader should score, not king
    state.board[9][0].tile = 'red'
    state.board[9][1].leader = { color: 'black', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'black')!.position = { row: 9, col: 1 }
    state.board[10][0].leader = { color: 'green', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'green')!.position = { row: 10, col: 0 }

    const result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 2 } })
    // Green leader scores +1 green VP, not black
    expect(result.players[0].score.green).toBe(1)
    expect(result.players[0].score.black).toBe(0)
  })
})
