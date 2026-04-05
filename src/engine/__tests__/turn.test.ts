import { createGame } from '../setup'
import { GameState, TileColor } from '../types'
import { endTurn, collectTreasures, checkGameEnd, calculateFinalScores } from '../turn'

const STARTING_TEMPLES: [number, number][] = [
  [0,10],[1,1],[1,15],[2,5],[4,13],[6,9],[7,1],[8,14],[9,6],[10,10],
]

function placeLeaderOnBoard(state: GameState, playerIndex: number, color: string, row: number, col: number) {
  const player = state.players[playerIndex]
  const leader = player.leaders.find(l => l.color === color)!
  leader.position = { row, col }
  state.board[row][col].leader = { color: color as any, dynasty: player.dynasty }
}

function clearCell(board: GameState['board'], row: number, col: number) {
  board[row][col].tile = null
  board[row][col].tileFlipped = false
  board[row][col].leader = null
  board[row][col].catastrophe = false
  board[row][col].monument = null
  board[row][col].hasTreasure = false
}

describe('turn flow', () => {
  it('draws tiles to refill hand to 6', () => {
    const state = createGame(2)
    // Simulate player used 2 tiles (hand has 4)
    state.players[0].hand = state.players[0].hand.slice(0, 4)
    state.currentPlayer = 0
    const bagBefore = state.bag.length

    const next = endTurn(state)
    expect(next.players[0].hand).toHaveLength(6)
    expect(next.bag).toHaveLength(bagBefore - 2)
  })

  it('treasure collected when kingdom has 2+ treasures and trader', () => {
    const state = createGame(2)
    // Clear area and build a kingdom with trader and 2 treasures
    // Use starting temples at [0,10] and [1,15] — too far apart
    // Instead, place tiles to connect two treasure cells and add a trader

    // Starting temple at [0,10] has treasure. Place a tile at [0,11] to extend.
    state.board[0][11].tile = 'green'
    state.board[0][11].hasTreasure = true // second treasure

    // Place green leader (trader) adjacent to a temple in this kingdom
    // Put trader at [1,10] — needs a tile or temple adjacent
    // [0,10] has a red temple tile already. Place trader at [1,10]
    state.board[1][10].tile = 'red' // temple-like tile so leader is adjacent to temple
    placeLeaderOnBoard(state, 0, 'green', 1, 11)
    // Connect: [0,10](treasure) - [0,11](treasure) - [1,11](leader needs tile adjacency)
    // Actually leader is at [1,11], let's connect properly
    // [0,10] temple+treasure, [0,11] tile+treasure, [1,11] leader
    // [1,11] is adjacent to [0,11] so they're connected

    const result = collectTreasures(state)
    // Should collect 1 treasure (2 treasures -> 1 remains)
    expect(result.players[0].treasures).toBe(1)
    // Count treasures in that kingdom area
    const t1 = result.board[0][10].hasTreasure
    const t2 = result.board[0][11].hasTreasure
    expect([t1, t2].filter(Boolean)).toHaveLength(1)
  })

  it('collects treasures from starting temples first', () => {
    const state = createGame(2)
    // Build kingdom with 3 treasures: starting temple [0,10], plus 2 non-temple treasures
    state.board[0][11].tile = 'green'
    state.board[0][11].hasTreasure = true
    state.board[0][12].tile = 'blue' // river cell but let's set it up on land
    // [0,12] is a river cell. Use [1,10] instead
    state.board[0][12].tile = null
    state.board[0][12].hasTreasure = false
    state.board[1][10].tile = 'blue'
    state.board[1][10].hasTreasure = true

    // Place trader at [1,11] to connect everything
    placeLeaderOnBoard(state, 0, 'green', 1, 11)
    // Kingdom: [0,10](temple+treasure), [0,11](treasure), [1,10](treasure), [1,11](leader)
    // Adjacency: [0,10]-[0,11], [0,10]-[1,10], [0,11]-[1,11], [1,10]-[1,11]

    const result = collectTreasures(state)
    // 3 treasures, collect 2, starting temple [0,10] collected FIRST
    expect(result.players[0].treasures).toBe(2)
    // Starting temple treasure at [0,10] should be collected first
    expect(result.board[0][10].hasTreasure).toBe(false)
    // One treasure remains
    const remaining = [
      result.board[0][11].hasTreasure,
      result.board[1][10].hasTreasure,
    ].filter(Boolean).length
    expect(remaining).toBe(1)
  })

  it('no collection without trader', () => {
    const state = createGame(2)
    // Kingdom with 2 treasures but no green leader
    state.board[0][11].tile = 'red'
    state.board[0][11].hasTreasure = true
    // [0,10] already has treasure from starting temple
    // Place a non-green leader
    placeLeaderOnBoard(state, 0, 'red', 1, 10)
    state.board[1][10].tile = 'red'

    const result = collectTreasures(state)
    expect(result.players[0].treasures).toBe(0)
    expect(result.board[0][10].hasTreasure).toBe(true)
    expect(result.board[0][11].hasTreasure).toBe(true)
  })

  it('game ends when ≤2 treasures remain on board', () => {
    const state = createGame(2)
    // Remove all but 2 treasures
    for (const [r, c] of STARTING_TEMPLES) {
      state.board[r][c].hasTreasure = false
    }
    state.board[0][10].hasTreasure = true
    state.board[1][1].hasTreasure = true
    expect(checkGameEnd(state)).toBe(true)
  })

  it('does not end game when 3+ treasures remain', () => {
    const state = createGame(2)
    // 10 treasures by default
    expect(checkGameEnd(state)).toBe(false)
  })

  it('game ends when bag is empty and player needs to draw', () => {
    const state = createGame(2)
    state.bag = []
    // Player has fewer than 6 tiles
    state.players[0].hand = ['red', 'blue']
    expect(checkGameEnd(state)).toBe(true)
  })

  it('does not end when bag is empty but hand is full', () => {
    const state = createGame(2)
    state.bag = []
    // All players have 6 tiles
    state.players[0].hand = ['red', 'blue', 'green', 'black', 'red', 'blue']
    state.players[1].hand = ['red', 'blue', 'green', 'black', 'red', 'blue']
    // Current player has full hand, so no need to draw
    expect(checkGameEnd(state)).toBe(false)
  })

  it('advances to next player and resets actions', () => {
    const state = createGame(3)
    state.currentPlayer = 0

    const next = endTurn(state)
    expect(next.currentPlayer).toBe(1)
    expect(next.actionsRemaining).toBe(2)
    expect(next.turnPhase).toBe('action')
  })

  it('wraps around player order', () => {
    const state = createGame(3)
    state.currentPlayer = 2

    const next = endTurn(state)
    expect(next.currentPlayer).toBe(0)
    expect(next.actionsRemaining).toBe(2)
  })

  it('sets gameOver when game should end', () => {
    const state = createGame(2)
    // Remove all but 2 treasures
    for (const [r, c] of STARTING_TEMPLES) {
      state.board[r][c].hasTreasure = false
    }
    state.board[0][10].hasTreasure = true
    state.board[1][1].hasTreasure = true

    const next = endTurn(state)
    expect(next.turnPhase).toBe('gameOver')
  })
})

describe('final scoring', () => {
  it('score is minimum across 4 colors', () => {
    const state = createGame(2)
    state.players[0].score = { red: 5, blue: 8, green: 3, black: 7 }
    state.players[0].treasures = 0
    state.players[1].score = { red: 2, blue: 2, green: 2, black: 2 }
    state.players[1].treasures = 0

    const scores = calculateFinalScores(state)
    const p0 = scores.find(s => s.playerIndex === 0)!
    expect(p0.finalScore).toBe(3) // min of 5,8,3,7
    const p1 = scores.find(s => s.playerIndex === 1)!
    expect(p1.finalScore).toBe(2)
  })

  it('treasures assigned to weakest color', () => {
    const state = createGame(2)
    state.players[0].score = { red: 5, blue: 8, green: 3, black: 7 }
    state.players[0].treasures = 2
    state.players[1].score = { red: 0, blue: 0, green: 0, black: 0 }
    state.players[1].treasures = 0

    const scores = calculateFinalScores(state)
    const p0 = scores.find(s => s.playerIndex === 0)!
    // green=3 is weakest, +2 treasures -> green=5, now min is red=5
    expect(p0.finalScore).toBe(5)
    expect(p0.colorScores.green).toBe(5)
  })

  it('treasures spread across multiple weak colors', () => {
    const state = createGame(2)
    state.players[0].score = { red: 1, blue: 1, green: 1, black: 10 }
    state.players[0].treasures = 3
    state.players[1].score = { red: 0, blue: 0, green: 0, black: 0 }

    const scores = calculateFinalScores(state)
    const p0 = scores.find(s => s.playerIndex === 0)!
    // 3 treasures: each of red, blue, green gets +1 -> all become 2, min=2
    expect(p0.finalScore).toBe(2)
    expect(p0.colorScores.red).toBe(2)
    expect(p0.colorScores.blue).toBe(2)
    expect(p0.colorScores.green).toBe(2)
    expect(p0.colorScores.black).toBe(10)
  })

  it('tiebreak by second-lowest, third, fourth', () => {
    const state = createGame(2)
    // Both have min=3, but p0 has higher second-lowest
    state.players[0].score = { red: 3, blue: 5, green: 6, black: 7 }
    state.players[0].treasures = 0
    state.players[1].score = { red: 3, blue: 4, green: 6, black: 8 }
    state.players[1].treasures = 0

    const scores = calculateFinalScores(state)
    // Both have min 3, but p0 second-lowest=5 > p1 second-lowest=4
    // So p0 should be ranked higher (come first if sorted by score desc)
    // The function returns sorted by score descending
    expect(scores[0].playerIndex).toBe(0)
    expect(scores[1].playerIndex).toBe(1)
  })
})
