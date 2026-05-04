import { describe, it, expect } from 'vitest'
import { findCatastropheMove, getAIAction } from '../simpleAI'
import { createGame } from '../../engine/setup'
import { GameState, Position, TileColor, LeaderColor, Dynasty } from '../../engine/types'

/**
 * Build a game with a totally blank board (no starting temples/treasures) so
 * tests can place exact patterns without colliding with the default temple
 * layout (which has treasures and would block catastrophe placements).
 */
function blankBoardGame(): GameState {
  const state = createGame(2, [false, true])
  for (let r = 0; r < state.board.length; r++) {
    for (let c = 0; c < state.board[0].length; c++) {
      const cell = state.board[r][c]
      cell.tile = null
      cell.tileFlipped = false
      cell.leader = null
      cell.catastrophe = false
      cell.monument = null
      cell.hasTreasure = false
    }
  }
  // AI is player 1.
  state.currentPlayer = 1
  return state
}

function placeTile(state: GameState, pos: Position, color: TileColor) {
  state.board[pos.row][pos.col].tile = color
  state.board[pos.row][pos.col].tileFlipped = false
}

function placeLeader(
  state: GameState,
  playerIndex: number,
  color: LeaderColor,
  pos: Position,
) {
  const dynasty: Dynasty = state.players[playerIndex].dynasty
  state.board[pos.row][pos.col].leader = { color, dynasty }
  const leaderEntry = state.players[playerIndex].leaders.find(l => l.color === color)!
  leaderEntry.position = pos
}

describe('findCatastropheMove — pattern 1: sever opponent leader from temples', () => {
  it('fires when opponent leader is in a kingdom with exactly one red temple and opponent leads in that color', () => {
    const state = blankBoardGame()
    // Build a small kingdom containing player 0's red leader plus a single red temple.
    // Layout (row,col):
    //   (5,5) red tile (temple)
    //   (5,6) opponent red leader
    //   (5,7) green tile (with treasure — satisfies tightened pattern-1 guard b)
    placeTile(state, { row: 5, col: 5 }, 'red')
    placeLeader(state, 0, 'red', { row: 5, col: 6 })
    placeTile(state, { row: 5, col: 7 }, 'green')
    state.board[5][7].hasTreasure = true

    // Opponent (player 0) is ahead in red; AI (player 1) is behind.
    state.players[0].score.red = 5
    state.players[1].score.red = 1

    const action = findCatastropheMove(state, 1)
    expect(action).not.toBeNull()
    if (action && action.type === 'placeCatastrophe') {
      expect(action.position).toEqual({ row: 5, col: 5 })
    } else {
      throw new Error(`expected placeCatastrophe, got ${action?.type ?? 'null'}`)
    }
  })

  it('does NOT fire when AI is ahead in the affected color (asymmetric guard)', () => {
    const state = blankBoardGame()
    placeTile(state, { row: 5, col: 5 }, 'red')
    placeLeader(state, 0, 'red', { row: 5, col: 6 })
    placeTile(state, { row: 5, col: 7 }, 'green')

    // Flip score asymmetry: AI is ahead in red, so we shouldn't bother disrupting.
    state.players[0].score.red = 1
    state.players[1].score.red = 5

    const action = findCatastropheMove(state, 1)
    expect(action).toBeNull()
  })
})

describe('findCatastropheMove — pattern 2: split a large opponent kingdom', () => {
  it('fires on an articulation point that bridges two halves with 2 opponent leaders of different colors', () => {
    const state = blankBoardGame()
    // Build a 7-tile kingdom shaped as two clusters joined by a single bridge tile.
    // Left cluster (3 tiles + 1 leader): (3,3)=red tile, (4,3)=red tile, (5,3)=green leader.
    // Bridge:                            (5,4)=green tile (the cut vertex).
    // Right cluster (3 tiles + 1 leader): (5,5)=blue tile, (4,5)=red leader, (3,5)=red tile.
    placeTile(state, { row: 3, col: 3 }, 'red')
    placeTile(state, { row: 4, col: 3 }, 'red')
    placeLeader(state, 0, 'green', { row: 5, col: 3 })
    placeTile(state, { row: 5, col: 4 }, 'green') // bridge
    placeTile(state, { row: 5, col: 5 }, 'blue')
    placeLeader(state, 0, 'red', { row: 4, col: 5 })
    placeTile(state, { row: 3, col: 5 }, 'red')

    // Opponent has 2 leaders here -> their highest-scoring kingdom heuristic.
    const action = findCatastropheMove(state, 1)
    expect(action).not.toBeNull()
    if (action && action.type === 'placeCatastrophe') {
      // Multiple articulation points exist in this layout: (4,3), (5,4), (5,5).
      // Any of them is a valid kingdom-split target.
      const valid = ['4,3', '5,4', '5,5']
      const key = `${action.position.row},${action.position.col}`
      expect(valid).toContain(key)
    } else {
      throw new Error(`expected placeCatastrophe, got ${action?.type ?? 'null'}`)
    }
  })
})

describe('findCatastropheMove — pattern 3: block opponent monument formation', () => {
  it('catastrophes one of three same-color tiles forming an L that would complete a 2x2', () => {
    const state = blankBoardGame()
    // 3-of-4 same-color L shape at (5,5)/(5,6)/(6,5). Empty hole is (6,6).
    // Anchor opponent leader to the kingdom via TWO red temples so pattern 1
    // (sever-leader) doesn't outrank pattern 3.
    placeTile(state, { row: 5, col: 5 }, 'green')
    placeTile(state, { row: 5, col: 6 }, 'green')
    placeTile(state, { row: 6, col: 5 }, 'green')
    placeTile(state, { row: 4, col: 5 }, 'red')
    placeTile(state, { row: 4, col: 7 }, 'red')
    placeLeader(state, 0, 'green', { row: 4, col: 6 })

    // Opponent is ahead in green.
    state.players[0].score.green = 5
    state.players[1].score.green = 1

    const action = findCatastropheMove(state, 1)
    expect(action).not.toBeNull()
    if (action && action.type === 'placeCatastrophe') {
      const targets = ['5,5', '5,6', '6,5']
      const key = `${action.position.row},${action.position.col}`
      expect(targets).toContain(key)
    } else {
      throw new Error(`expected placeCatastrophe, got ${action?.type ?? 'null'}`)
    }
  })
})

describe('findCatastropheMove — guards', () => {
  it('returns null when AI has no catastrophes remaining', () => {
    const state = blankBoardGame()
    // Set up pattern 1 (would otherwise fire).
    placeTile(state, { row: 5, col: 5 }, 'red')
    placeLeader(state, 0, 'red', { row: 5, col: 6 })
    placeTile(state, { row: 5, col: 7 }, 'green')
    state.players[0].score.red = 5
    state.players[1].score.red = 1

    state.players[1].catastrophesRemaining = 0
    const action = findCatastropheMove(state, 1)
    expect(action).toBeNull()
  })
})

describe('findCatastropheMove — regression on fresh game', () => {
  it('returns null on a fresh-game state (no triggers have fired)', () => {
    const state = createGame(2, [false, true])
    state.currentPlayer = 1
    const action = findCatastropheMove(state, 1)
    expect(action).toBeNull()
  })

  it('AI overall first move on a fresh game is unchanged (still placeLeader)', () => {
    const state = createGame(2, [false, true])
    state.currentPlayer = 1
    const action = getAIAction(state)
    expect(action.type).toBe('placeLeader')
  })
})

describe('findCatastropheMove — pattern 1 tightened guards', () => {
  // Guard (a): score margin must be >= 2 (not just > 0).
  it('does NOT fire when opponent leads in the affected color by exactly 1 (margin guard)', () => {
    const state = blankBoardGame()
    // Standard pattern-1 setup, with treasure present so guard (b) is satisfied.
    placeTile(state, { row: 5, col: 5 }, 'red')
    placeLeader(state, 0, 'red', { row: 5, col: 6 })
    placeTile(state, { row: 5, col: 7 }, 'green')
    state.board[5][7].hasTreasure = true

    // Margin = 1 (opp 3, AI 2). Should NOT fire.
    state.players[0].score.red = 3
    state.players[1].score.red = 2

    const action = findCatastropheMove(state, 1)
    expect(action).toBeNull()
  })

  it('FIRES when opponent leads in the affected color by exactly 2 (margin guard satisfied)', () => {
    const state = blankBoardGame()
    placeTile(state, { row: 5, col: 5 }, 'red')
    placeLeader(state, 0, 'red', { row: 5, col: 6 })
    placeTile(state, { row: 5, col: 7 }, 'green')
    state.board[5][7].hasTreasure = true

    state.players[0].score.red = 4
    state.players[1].score.red = 2

    const action = findCatastropheMove(state, 1)
    expect(action).not.toBeNull()
    if (action && action.type === 'placeCatastrophe') {
      expect(action.position).toEqual({ row: 5, col: 5 })
    } else {
      throw new Error(`expected placeCatastrophe, got ${action?.type ?? 'null'}`)
    }
  })

  // Guard (b): kingdom containing the leader must hold a treasure.
  it('does NOT fire when leader\'s kingdom holds no treasure (treasure guard)', () => {
    const state = blankBoardGame()
    placeTile(state, { row: 5, col: 5 }, 'red')
    placeLeader(state, 0, 'red', { row: 5, col: 6 })
    placeTile(state, { row: 5, col: 7 }, 'green')
    // No treasure in this kingdom.

    // Big margin so guard (a) is satisfied.
    state.players[0].score.red = 9
    state.players[1].score.red = 1

    const action = findCatastropheMove(state, 1)
    expect(action).toBeNull()
  })

  it('FIRES when leader\'s kingdom holds a treasure (treasure guard satisfied)', () => {
    const state = blankBoardGame()
    placeTile(state, { row: 5, col: 5 }, 'red')
    placeLeader(state, 0, 'red', { row: 5, col: 6 })
    placeTile(state, { row: 5, col: 7 }, 'green')
    state.board[5][7].hasTreasure = true

    state.players[0].score.red = 9
    state.players[1].score.red = 1

    const action = findCatastropheMove(state, 1)
    expect(action).not.toBeNull()
    if (action && action.type === 'placeCatastrophe') {
      expect(action.position).toEqual({ row: 5, col: 5 })
    } else {
      throw new Error(`expected placeCatastrophe, got ${action?.type ?? 'null'}`)
    }
  })

  // Composable AND: both guards must hold for pattern 1 to fire.
  it('fires when BOTH guards are satisfied (margin >= 2 AND treasure in kingdom)', () => {
    const state = blankBoardGame()
    placeTile(state, { row: 5, col: 5 }, 'red')
    placeLeader(state, 0, 'red', { row: 5, col: 6 })
    placeTile(state, { row: 5, col: 7 }, 'green')
    state.board[5][7].hasTreasure = true

    state.players[0].score.red = 5
    state.players[1].score.red = 2

    const action = findCatastropheMove(state, 1)
    expect(action).not.toBeNull()
    if (action && action.type === 'placeCatastrophe') {
      expect(action.position).toEqual({ row: 5, col: 5 })
    } else {
      throw new Error(`expected placeCatastrophe, got ${action?.type ?? 'null'}`)
    }
  })
})
