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

export function findConnectedGroup(board: Cell[][], start: Position): Position[] {
  const startCell = board[start.row][start.col]
  if (!isOccupied(startCell)) return []

  const visited = new Set<string>()
  const group: Position[] = []
  const queue: Position[] = [start]
  visited.add(`${start.row},${start.col}`)

  while (queue.length > 0) {
    const pos = queue.shift()!
    group.push(pos)

    for (const neighbor of getNeighbors(pos)) {
      const key = `${neighbor.row},${neighbor.col}`
      if (visited.has(key)) continue
      visited.add(key)
      const cell = board[neighbor.row][neighbor.col]
      if (isOccupied(cell)) {
        queue.push(neighbor)
      }
    }
  }

  return group
}

export function findKingdoms(board: Cell[][]): Kingdom[] {
  const visited = new Set<string>()
  const kingdoms: Kingdom[] = []

  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const key = `${row},${col}`
      if (visited.has(key)) continue
      const cell = board[row][col]
      if (!isOccupied(cell)) continue

      const group = findConnectedGroup(board, { row, col })
      for (const pos of group) {
        visited.add(`${pos.row},${pos.col}`)
      }

      const leaders: Kingdom['leaders'] = []
      for (const pos of group) {
        const c = board[pos.row][pos.col]
        if (c.leader) {
          leaders.push({
            color: c.leader.color,
            dynasty: c.leader.dynasty,
            position: pos,
          })
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
  const visited = new Set<string>()
  const regions: Position[][] = []

  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const key = `${row},${col}`
      if (visited.has(key)) continue
      const cell = board[row][col]
      if (!isOccupied(cell)) continue

      const group = findConnectedGroup(board, { row, col })
      for (const pos of group) {
        visited.add(`${pos.row},${pos.col}`)
      }

      const hasLeader = group.some(
        (pos) => board[pos.row][pos.col].leader !== null
      )

      if (!hasLeader) {
        regions.push(group)
      }
    }
  }

  return regions
}
