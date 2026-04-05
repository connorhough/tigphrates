import { createInitialBoard, BOARD_ROWS, BOARD_COLS } from '../types'

describe('types', () => {
  it('creates a board with correct dimensions', () => {
    const board = createInitialBoard()
    expect(board).toHaveLength(BOARD_ROWS)
    expect(board[0]).toHaveLength(BOARD_COLS)
  })

  it('marks river cells correctly', () => {
    const board = createInitialBoard()
    // Row 0, col 4 should be river
    expect(board[0][4].terrain).toBe('river')
    // Row 0, col 0 should be land
    expect(board[0][0].terrain).toBe('land')
  })

  it('places starting temples with treasures', () => {
    const board = createInitialBoard()
    // (col 10, row 0) is a starting temple
    expect(board[0][10].tile).toBe('red')
    expect(board[0][10].hasTreasure).toBe(true)
    // (col 0, row 0) is empty land
    expect(board[0][0].tile).toBeNull()
    expect(board[0][0].hasTreasure).toBe(false)
  })
})
