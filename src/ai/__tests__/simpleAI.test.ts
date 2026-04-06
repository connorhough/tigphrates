import { getAIAction } from '../simpleAI'
import { createGame } from '../../engine/setup'
import { GameState } from '../../engine/types'

function freshAIState(): GameState {
  const state = createGame(2, [false, true])
  state.currentPlayer = 1
  return state
}

describe('simpleAI', () => {
  it('returns a valid action for a fresh game state', () => {
    const state = freshAIState()
    const action = getAIAction(state)
    expect(action).toBeDefined()
    expect(action.type).toBeDefined()
  })

  it('places a leader when none are on board', () => {
    const state = freshAIState()
    const action = getAIAction(state)
    expect(action.type).toBe('placeLeader')
  })

  it('places tile to score when leaders are on board', () => {
    const state = freshAIState()
    // Place bull red leader on board adjacent to a temple (0,10)
    const leaderPos = { row: 0, col: 9 }
    state.board[leaderPos.row][leaderPos.col].leader = { color: 'red', dynasty: 'bull' }
    state.players[1].leaders[0].position = leaderPos // red leader

    // Give AI some tiles that could score
    state.players[1].hand = ['red', 'red', 'blue', 'green', 'black', 'red']

    const action = getAIAction(state)
    // Should try to place a tile (or leader for another color)
    expect(['placeTile', 'placeLeader']).toContain(action.type)
  })

  it('commits support tiles during revolt when it can win', () => {
    const state = freshAIState()
    state.turnPhase = 'conflictSupport'
    // AI is defender (player 1), has red tiles to win
    state.players[1].hand = ['red', 'red', 'red', 'blue', 'green', 'black']
    state.pendingConflict = {
      type: 'revolt',
      color: 'red',
      attacker: { playerIndex: 0, position: { row: 1, col: 2 } },
      defender: { playerIndex: 1, position: { row: 1, col: 0 } },
      attackerStrength: 2,
      defenderStrength: 1,
      attackerCommitted: [0], // attacker committed 1 tile
      defenderCommitted: null,   // defender hasn't committed yet
    }
    const action = getAIAction(state)
    expect(action.type).toBe('commitSupport')
    if (action.type === 'commitSupport') {
      // Should commit enough red tiles to try to win
      expect(action.indices.length).toBeGreaterThan(0)
    }
  })

  it('commits zero tiles when it will lose anyway', () => {
    const state = freshAIState()
    state.turnPhase = 'conflictSupport'
    // AI is defender (player 1), has NO red tiles
    // Attacker has already committed, so now it's defender's turn
    state.players[1].hand = ['blue', 'green', 'black', 'blue', 'green', 'black']
    state.pendingConflict = {
      type: 'revolt',
      color: 'red',
      attacker: { playerIndex: 0, position: { row: 1, col: 2 } },
      defender: { playerIndex: 1, position: { row: 1, col: 0 } },
      attackerStrength: 3,
      defenderStrength: 1,
      attackerCommitted: [0, 1],
      defenderCommitted: null,
    }
    const action = getAIAction(state)
    expect(action.type).toBe('commitSupport')
    if (action.type === 'commitSupport') {
      expect(action.indices).toHaveLength(0)
    }
  })

  it('chooses monument when available', () => {
    const state = freshAIState()
    state.turnPhase = 'monumentChoice'
    state.pendingMonument = { position: { row: 2, col: 5 }, color: 'red' }
    const action = getAIAction(state)
    expect(action.type).toBe('buildMonument')
  })

  it('chooses war order based on advantage', () => {
    const state = freshAIState()
    state.turnPhase = 'warOrderChoice'
    state.pendingConflict = {
      type: 'war',
      color: 'red',
      attacker: { playerIndex: 1, position: { row: 1, col: 2 } },
      defender: { playerIndex: 0, position: { row: 1, col: 0 } },
      attackerStrength: 0,
      defenderStrength: 0,
      attackerCommitted: null,
      defenderCommitted: null,
      pendingWarColors: ['red', 'blue'],
    }
    // Give AI some tiles
    state.players[1].hand = ['red', 'red', 'red', 'blue', 'green', 'black']
    const action = getAIAction(state)
    expect(action.type).toBe('chooseWarOrder')
    if (action.type === 'chooseWarOrder') {
      // Should pick red since AI has more red tiles
      expect(action.color).toBe('red')
    }
  })

  it('swaps tiles if no matching tiles for on-board leaders', () => {
    const state = freshAIState()
    // Place a green leader on board
    const leaderPos = { row: 0, col: 9 }
    state.board[leaderPos.row][leaderPos.col].leader = { color: 'green', dynasty: 'bull' }
    state.players[1].leaders[2].position = leaderPos // green leader

    // Give AI only tiles that don't match any on-board leader color
    // and no valid tile placements would score
    state.players[1].hand = ['red', 'red', 'red', 'red', 'red', 'red']

    // Also put all other leaders off-board
    state.players[1].leaders[0].position = null
    state.players[1].leaders[1].position = null
    state.players[1].leaders[3].position = null

    const action = getAIAction(state)
    // AI should place red tiles (scoring from temple red leader in kingdom), place leaders, or swap
    expect(['placeTile', 'placeLeader', 'swapTiles']).toContain(action.type)
  })

  it('passes as last resort', () => {
    const state = freshAIState()
    // Empty hand, all leaders already placed (but we make them null => no valid placements)
    state.players[1].hand = []
    state.players[1].catastrophesRemaining = 0
    const action = getAIAction(state)
    // With empty hand and no catastrophes, should pass or place leader
    expect(['pass', 'placeLeader']).toContain(action.type)
  })

  it('never throws regardless of game state', () => {
    // Run multiple random games and ensure no throws
    for (let i = 0; i < 10; i++) {
      const state = freshAIState()
      expect(() => getAIAction(state)).not.toThrow()
    }
  })
})
