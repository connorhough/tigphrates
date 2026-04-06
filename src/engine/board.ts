import { Position, Cell, BOARD_ROWS, BOARD_COLS, LeaderColor, Dynasty } from './types'

export interface Kingdom {
  positions: Position[]
  leaders: { color: LeaderColor; dynasty: Dynasty; position: Position }[]
}

export function getNeighbors(pos: Position): Position[] {
  const { row, col } = pos
  const neighbors: Position[] = []
  if (row > 0) neighbors.push({ row: row - 1, col })
  if (row < BOARD_ROWS - 1) neighbors.push({ row: row + 1, col })
  if (col > 0) neighbors.push({ row, col: col - 1 })
  if (col < BOARD_COLS - 1) neighbors.push({ row, col: col + 1 })
  return neighbors
}

function isOccupied(cell: Cell): boolean {
  return !cell.catastrophe && (cell.tile !== null || cell.leader !== null)
}

// Module-level reusable BFS buffers — avoids per-call allocations
const _vis = new Uint8Array(BOARD_ROWS * BOARD_COLS)
const _q = new Int32Array(BOARD_ROWS * BOARD_COLS)

/**
 * Internal BFS flood fill. Does NOT reset _vis — caller is responsible.
 * Marks all reachable occupied cells starting from startIdx.
 * Returns the group as Position[].
 */
function _bfs(board: Cell[][], startIdx: number): Position[] {
  let head = 0
  let tail = 0
  _q[tail++] = startIdx
  _vis[startIdx] = 1
  const group: Position[] = []

  while (head < tail) {
    const idx = _q[head++]
    const r = (idx / BOARD_COLS) | 0
    const c = idx % BOARD_COLS
    group.push({ row: r, col: c })

    // up
    if (r > 0) {
      const n = idx - BOARD_COLS
      if (!_vis[n] && isOccupied(board[r - 1][c])) { _vis[n] = 1; _q[tail++] = n }
    }
    // down
    if (r < BOARD_ROWS - 1) {
      const n = idx + BOARD_COLS
      if (!_vis[n] && isOccupied(board[r + 1][c])) { _vis[n] = 1; _q[tail++] = n }
    }
    // left
    if (c > 0) {
      const n = idx - 1
      if (!_vis[n] && isOccupied(board[r][c - 1])) { _vis[n] = 1; _q[tail++] = n }
    }
    // right
    if (c < BOARD_COLS - 1) {
      const n = idx + 1
      if (!_vis[n] && isOccupied(board[r][c + 1])) { _vis[n] = 1; _q[tail++] = n }
    }
  }

  return group
}

export function findConnectedGroup(board: Cell[][], start: Position): Position[] {
  const startIdx = start.row * BOARD_COLS + start.col
  if (!isOccupied(board[start.row][start.col])) return []
  _vis.fill(0)
  return _bfs(board, startIdx)
}

export function findKingdoms(board: Cell[][]): Kingdom[] {
  _vis.fill(0)
  const kingdoms: Kingdom[] = []

  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const idx = row * BOARD_COLS + col
      if (_vis[idx]) continue
      if (!isOccupied(board[row][col])) continue

      // _bfs marks all cells in this group in _vis
      const group = _bfs(board, idx)

      const leaders: Kingdom['leaders'] = []
      for (const pos of group) {
        const cell = board[pos.row][pos.col]
        if (cell.leader) {
          leaders.push({ color: cell.leader.color, dynasty: cell.leader.dynasty, position: pos })
        }
      }

      if (leaders.length > 0) {
        kingdoms.push({ positions: group, leaders })
      }
    }
  }

  return kingdoms
}

export function findRegions(board: Cell[][]): Position[][] {
  _vis.fill(0)
  const regions: Position[][] = []

  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const idx = row * BOARD_COLS + col
      if (_vis[idx]) continue
      if (!isOccupied(board[row][col])) continue

      const group = _bfs(board, idx)

      const hasLeader = group.some(pos => board[pos.row][pos.col].leader !== null)
      if (!hasLeader) {
        regions.push(group)
      }
    }
  }

  return regions
}
