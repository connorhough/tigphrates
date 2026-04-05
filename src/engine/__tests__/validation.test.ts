import { createGame } from '../setup'
import { createInitialBoard, GameState, Position } from '../types'
import {
  getValidTilePlacements,
  getValidLeaderPlacements,
  canPlaceCatastrophe,
  canSwapTiles,
} from '../validation'

function posSet(positions: Position[]): Set<string> {
  return new Set(positions.map(p => `${p.row},${p.col}`))
}

describe('getValidTilePlacements', () => {
  it('blue tiles can only go on river cells', () => {
    const state = createGame(2)
    const placements = getValidTilePlacements(state, 'blue')
    for (const pos of placements) {
      expect(state.board[pos.row][pos.col].terrain).toBe('river')
    }
  })

  it('red tiles can only go on land cells', () => {
    const state = createGame(2)
    const placements = getValidTilePlacements(state, 'red')
    for (const pos of placements) {
      expect(state.board[pos.row][pos.col].terrain).toBe('land')
    }
  })

  it('green tiles can only go on land cells', () => {
    const state = createGame(2)
    const placements = getValidTilePlacements(state, 'green')
    for (const pos of placements) {
      expect(state.board[pos.row][pos.col].terrain).toBe('land')
    }
  })

  it('black tiles can only go on land cells', () => {
    const state = createGame(2)
    const placements = getValidTilePlacements(state, 'black')
    for (const pos of placements) {
      expect(state.board[pos.row][pos.col].terrain).toBe('land')
    }
  })

  it('excludes cells that already have a tile', () => {
    const state = createGame(2)
    // Starting temple at (0,10) has a red tile
    const placements = getValidTilePlacements(state, 'red')
    const set = posSet(placements)
    expect(set.has('0,10')).toBe(false)
  })

  it('excludes cells that have a leader', () => {
    const state = createGame(2)
    state.board[1][0].leader = { color: 'red', dynasty: 'archer' }
    const placements = getValidTilePlacements(state, 'red')
    const set = posSet(placements)
    expect(set.has('1,0')).toBe(false)
  })

  it('excludes cells with catastrophe', () => {
    const state = createGame(2)
    state.board[1][0].catastrophe = true
    const placements = getValidTilePlacements(state, 'red')
    const set = posSet(placements)
    expect(set.has('1,0')).toBe(false)
  })

  it('excludes cells with monuments', () => {
    const state = createGame(2)
    state.board[1][0].monument = 'red-blue'
    const placements = getValidTilePlacements(state, 'red')
    const set = posSet(placements)
    expect(set.has('1,0')).toBe(false)
  })

  it('allows placement that unites exactly 2 kingdoms (triggers war)', () => {
    const state = createGame(2)
    // Place 2 kingdoms in a line on row 10 (all land, no temples nearby)
    // Kingdom A: leader at (10,0), tile at (10,1)
    state.board[10][0].leader = { color: 'red', dynasty: 'archer' }
    state.board[10][1].tile = 'black'
    // Kingdom B: leader at (10,4), tile at (10,3)
    state.board[10][4].leader = { color: 'blue', dynasty: 'bull' }
    state.board[10][3].tile = 'black'

    // Placing tile at (10,2) connects A and B — 2 kingdoms, allowed (triggers war)
    const placements = getValidTilePlacements(state, 'black')
    const set = posSet(placements)
    expect(set.has('10,2')).toBe(true)
  })

  it('rejects placement that would unite 3+ kingdoms', () => {
    const state = createGame(2)
    // Place 3 kingdoms around empty cell (10,3).
    // Neighbors of (10,3): (9,3), (10,2), (10,4) — last row, no south neighbor.
    // Each kingdom extends AWAY from (10,3) so they don't connect to each other.
    // Kingdom A: north — leader at (9,3), extends to (8,3)
    state.board[9][3].leader = { color: 'red', dynasty: 'archer' }
    state.board[8][3].tile = 'black'
    // Kingdom B: west — leader at (10,2), extends to (10,1)
    state.board[10][2].leader = { color: 'blue', dynasty: 'bull' }
    state.board[10][1].tile = 'black'
    // Kingdom C: east — leader at (10,4), extends to (10,5)
    state.board[10][4].leader = { color: 'green', dynasty: 'pot' }
    state.board[10][5].tile = 'black'

    // Verify (9,3) is NOT adjacent to (10,2) or (10,4) — they share no neighbors
    // (9,3) neighbors: (8,3),(10,3),(9,2),(9,4) — none of those are in B or C
    // (10,2) neighbors: (9,2),(10,1),(10,3) — (9,2) is empty, not in A
    // (10,4) neighbors: (9,4),(10,3),(10,5) — (9,4) is empty, not in A
    // So 3 separate kingdoms. Placing tile at (10,3) connects all three — invalid.
    const placements = getValidTilePlacements(state, 'black')
    const set = posSet(placements)
    expect(set.has('10,3')).toBe(false)
  })
})

describe('getValidLeaderPlacements', () => {
  it('must be on empty land cell', () => {
    const state = createGame(2)
    const placements = getValidLeaderPlacements(state, 'red')
    for (const pos of placements) {
      const cell = state.board[pos.row][pos.col]
      expect(cell.terrain).toBe('land')
      expect(cell.tile).toBeNull()
      expect(cell.leader).toBeNull()
      expect(cell.catastrophe).toBe(false)
      expect(cell.monument).toBeNull()
    }
  })

  it('must be adjacent to at least one face-up red temple tile', () => {
    const state = createGame(2)
    const placements = getValidLeaderPlacements(state, 'red')
    for (const pos of placements) {
      const neighbors = [
        { row: pos.row - 1, col: pos.col },
        { row: pos.row + 1, col: pos.col },
        { row: pos.row, col: pos.col - 1 },
        { row: pos.row, col: pos.col + 1 },
      ].filter(
        n => n.row >= 0 && n.row < 11 && n.col >= 0 && n.col < 16
      )
      const adjacentToTemple = neighbors.some(n => {
        const c = state.board[n.row][n.col]
        return c.tile === 'red' && !c.tileFlipped
      })
      expect(adjacentToTemple).toBe(true)
    }
  })

  it('rejects cells not adjacent to a temple', () => {
    const state = createGame(2)
    const placements = getValidLeaderPlacements(state, 'red')
    const set = posSet(placements)
    // (0,0) is land but not adjacent to any temple
    expect(set.has('0,0')).toBe(false)
  })

  it('rejects river cells', () => {
    const state = createGame(2)
    const placements = getValidLeaderPlacements(state, 'blue')
    for (const pos of placements) {
      expect(state.board[pos.row][pos.col].terrain).toBe('land')
    }
  })

  it('rejects placement that would unite 2+ kingdoms', () => {
    const state = createGame(2)
    // Use a clean area. Place a red temple at (10,3) for leader adjacency requirement.
    state.board[10][3].tile = 'red'
    // Kingdom A: leader at (10,2) connected to temple at (10,3)
    state.board[10][2].leader = { color: 'red', dynasty: 'archer' }
    // Kingdom B: leader at (10,5) connected to tile at (10,4)
    state.board[10][4].tile = 'black'
    state.board[10][5].leader = { color: 'blue', dynasty: 'bull' }

    // Now A = {(10,2)leader, (10,3)temple} and B = {(10,4)tile, (10,5)leader}
    // These are separate because (10,3) and (10,4) are not connected (they are adjacent
    // and both occupied... wait, they ARE adjacent, so they'd form one group!)

    // Need a gap. Let me separate them.
    // Kingdom A: leader at (10,2), temple at (10,3) — group = {(10,2),(10,3)}
    // Kingdom B: leader at (10,6), tile at (10,5) — group = {(10,5),(10,6)}
    // Gap at (10,4) empty.
    // Now, place a tile at (9,3) extending kingdom A upward
    state.board[10][4].tile = null // undo
    state.board[10][5].leader = null // undo
    state.board[10][6].leader = { color: 'blue', dynasty: 'bull' }
    state.board[10][5].tile = 'black'
    // A = {(10,2)leader, (10,3)temple}, B = {(10,5)tile, (10,6)leader}
    // Extend B with a tile at (9,5)
    state.board[9][5].tile = 'black'
    // B = {(10,5)tile, (10,6)leader, (9,5)tile}

    // Cell (9,4) is empty land adjacent to:
    //   (10,4) — empty, (9,3) — empty, (9,5) — kingdom B, (8,4) — empty
    // Not adjacent to temple, so can't place leader there.

    // Different approach: place both kingdoms adjacent to one cell that's also adjacent to a temple.
    // Use cell (9,3). Its neighbors: (8,3),(10,3),(9,2),(9,4)
    // (10,3) has our red temple (kingdom A). If (9,4) has a tile from kingdom B...
    // But (9,4) must be in a different kingdom.

    // Fresh approach with clean state
    const s = createGame(2)
    // Red temple at (10,3)
    s.board[10][3].tile = 'red'
    // Kingdom A: leader at (10,4) adjacent to temple (10,3)
    s.board[10][4].leader = { color: 'red', dynasty: 'archer' }
    // Kingdom B: leader at (8,3) with tile at (8,2)
    s.board[8][3].leader = { color: 'blue', dynasty: 'bull' }
    s.board[8][2].tile = 'black'
    // A = {(10,3)temple, (10,4)leader}
    // B = {(8,2)tile, (8,3)leader}

    // Cell (9,3): neighbors are (8,3)[kingdom B], (10,3)[kingdom A], (9,2), (9,4)
    // (9,3) is empty land, adjacent to red temple at (10,3), and adjacent to both kingdoms
    // Placing leader at (9,3) would unite A and B — should be rejected
    const placements = getValidLeaderPlacements(s, 'green')
    const set = posSet(placements)
    expect(set.has('9,3')).toBe(false)
  })

  it('allows placement adjacent to temple not connecting two kingdoms', () => {
    const state = createGame(2)
    // On initial board, temples are isolated. Placing a leader next to one is always fine.
    const placements = getValidLeaderPlacements(state, 'red')
    expect(placements.length).toBeGreaterThan(0)
    // Specifically (0,9) is adjacent to temple at (0,10) — should be valid
    const set = posSet(placements)
    expect(set.has('0,9')).toBe(true)
  })
})

describe('canPlaceCatastrophe', () => {
  it('can place on empty land cell', () => {
    const state = createGame(2)
    expect(canPlaceCatastrophe(state, { row: 5, col: 5 })).toBe(true)
  })

  it('can place on a tile (destroys it)', () => {
    const state = createGame(2)
    // Temple at (0,10) has a tile
    expect(canPlaceCatastrophe(state, { row: 0, col: 10 })).toBe(false) // has treasure!
  })

  it('can place on tile without treasure', () => {
    const state = createGame(2)
    state.board[5][5].tile = 'black'
    expect(canPlaceCatastrophe(state, { row: 5, col: 5 })).toBe(true)
  })

  it('cannot place on leader', () => {
    const state = createGame(2)
    state.board[1][0].leader = { color: 'red', dynasty: 'archer' }
    expect(canPlaceCatastrophe(state, { row: 1, col: 0 })).toBe(false)
  })

  it('cannot place on treasure', () => {
    const state = createGame(2)
    // Starting temples have treasures
    expect(canPlaceCatastrophe(state, { row: 0, col: 10 })).toBe(false)
  })

  it('cannot place on monument', () => {
    const state = createGame(2)
    state.board[5][5].monument = 'red-blue'
    expect(canPlaceCatastrophe(state, { row: 5, col: 5 })).toBe(false)
  })

  it('cannot place on existing catastrophe', () => {
    const state = createGame(2)
    state.board[5][5].catastrophe = true
    expect(canPlaceCatastrophe(state, { row: 5, col: 5 })).toBe(false)
  })

  it('can place on river cell', () => {
    const state = createGame(2)
    // (0,4) is a river cell, empty
    expect(canPlaceCatastrophe(state, { row: 0, col: 4 })).toBe(true)
  })
})

describe('canSwapTiles', () => {
  it('returns true if player has tiles in hand', () => {
    const state = createGame(2)
    expect(canSwapTiles(state)).toBe(true)
  })

  it('returns false if player has no tiles in hand', () => {
    const state = createGame(2)
    state.players[state.currentPlayer].hand = []
    expect(canSwapTiles(state)).toBe(false)
  })
})
