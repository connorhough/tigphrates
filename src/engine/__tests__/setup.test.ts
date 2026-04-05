import { createGame } from '../setup'

describe('createGame', () => {
  it('creates a 2-player game with correct initial state', () => {
    const state = createGame(2)
    expect(state.players).toHaveLength(2)
    expect(state.currentPlayer).toBe(0)
    expect(state.actionsRemaining).toBe(2)
    expect(state.turnPhase).toBe('action')
  })

  it('gives each player 6 tiles', () => {
    const state = createGame(2)
    for (const player of state.players) {
      expect(player.hand).toHaveLength(6)
    }
  })

  it('gives each player 4 leaders all off-board', () => {
    const state = createGame(2)
    for (const player of state.players) {
      expect(player.leaders).toHaveLength(4)
      for (const leader of player.leaders) {
        expect(leader.position).toBeNull()
      }
    }
  })

  it('gives each player 2 catastrophe tiles', () => {
    const state = createGame(2)
    for (const player of state.players) {
      expect(player.catastrophesRemaining).toBe(2)
    }
  })

  it('sets up the bag with correct total after dealing', () => {
    const state = createGame(2)
    // 153 total - 10 starting temples - 12 dealt (6 per player) = 131
    expect(state.bag.length + 10 + 12).toBe(153)
  })

  it('initializes all scores to zero', () => {
    const state = createGame(2)
    for (const player of state.players) {
      expect(player.score).toEqual({ red: 0, blue: 0, green: 0, black: 0 })
      expect(player.treasures).toBe(0)
    }
  })

  it('marks second player as AI when specified', () => {
    const state = createGame(2, [false, true])
    expect(state.players[0].isAI).toBe(false)
    expect(state.players[1].isAI).toBe(true)
  })

  it('initializes 6 available monuments', () => {
    const state = createGame(2)
    expect(state.monuments).toHaveLength(6)
    for (const m of state.monuments) {
      expect(m.position).toBeNull()
    }
  })
})
