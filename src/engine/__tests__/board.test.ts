import { getNeighbors, findConnectedGroup, findKingdoms, findRegions } from '../board'
import { createInitialBoard } from '../types'

describe('getNeighbors', () => {
  it('returns 4 neighbors for center cell', () => {
    expect(getNeighbors({ row: 5, col: 5 })).toHaveLength(4)
  })

  it('returns 2 neighbors for corner cell', () => {
    expect(getNeighbors({ row: 0, col: 0 })).toHaveLength(2)
  })

  it('returns 3 neighbors for edge cell', () => {
    expect(getNeighbors({ row: 0, col: 5 })).toHaveLength(3)
  })
})

describe('findConnectedGroup', () => {
  it('finds connected tiles from a starting position', () => {
    const board = createInitialBoard()
    // Starting temple at (row 0, col 10) is isolated — group of 1
    const group = findConnectedGroup(board, { row: 0, col: 10 })
    expect(group).toHaveLength(1)
  })
})

describe('findKingdoms', () => {
  it('returns empty array on initial board (no leaders)', () => {
    const board = createInitialBoard()
    const kingdoms = findKingdoms(board)
    expect(kingdoms).toHaveLength(0)
  })

  it('finds a kingdom when a leader is placed next to a temple', () => {
    const board = createInitialBoard()
    // Place a leader at (row 0, col 9) next to starting temple at (row 0, col 10)
    board[0][9].leader = { color: 'red', dynasty: 'archer' }
    const kingdoms = findKingdoms(board)
    expect(kingdoms).toHaveLength(1)
    expect(kingdoms[0].leaders).toHaveLength(1)
  })
})

describe('findRegions', () => {
  it('returns regions for initial board (all temples are isolated regions)', () => {
    const board = createInitialBoard()
    const regions = findRegions(board)
    // 10 starting temples, all isolated, so 10 regions
    expect(regions).toHaveLength(10)
  })

  it('returns no regions containing a leader', () => {
    const board = createInitialBoard()
    board[0][9].leader = { color: 'red', dynasty: 'archer' }
    const regions = findRegions(board)
    // The group with the leader becomes a kingdom, not a region
    // So we should have 9 regions (the other 9 isolated temples)
    expect(regions).toHaveLength(9)
  })
})
