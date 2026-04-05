import { GameState, Position, BOARD_ROWS, BOARD_COLS } from '../engine/types'
import { COLORS, DYNASTY_COLORS } from './colors'

export interface RenderConfig {
  cellSize: number
  highlights?: Position[]
  hoveredCell?: Position | null
  selectedCell?: Position | null
}

export function drawBoard(
  ctx: CanvasRenderingContext2D,
  state: GameState,
  config: RenderConfig,
): void {
  const { cellSize } = config
  const width = cellSize * BOARD_COLS
  const height = cellSize * BOARD_ROWS

  // 1. Clear canvas
  ctx.clearRect(0, 0, width, height)

  // 2. Draw terrain grid
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const cell = state.board[row][col]
      const x = col * cellSize
      const y = row * cellSize
      ctx.fillStyle = cell.terrain === 'river' ? COLORS.river : COLORS.land
      ctx.fillRect(x, y, cellSize, cellSize)
    }
  }

  // 3. Draw tiles (colored rectangles, slightly inset)
  const inset = 2
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const cell = state.board[row][col]
      if (!cell.tile) continue
      const x = col * cellSize + inset
      const y = row * cellSize + inset
      const size = cellSize - inset * 2

      if (cell.tileFlipped) {
        // 4. Face-down tiles: darker/muted version
        ctx.fillStyle = COLORS[cell.tile]
        ctx.globalAlpha = 0.4
        ctx.fillRect(x, y, size, size)
        ctx.globalAlpha = 1.0
      } else {
        ctx.fillStyle = COLORS[cell.tile]
        ctx.fillRect(x, y, size, size)
      }
    }
  }

  // 5. Draw leaders (colored circles with dynasty indicator)
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const cell = state.board[row][col]
      if (!cell.leader) continue
      const cx = col * cellSize + cellSize / 2
      const cy = row * cellSize + cellSize / 2
      const outerRadius = cellSize * 0.38
      const innerRadius = cellSize * 0.22

      // Outer circle: dynasty color
      ctx.beginPath()
      ctx.arc(cx, cy, outerRadius, 0, Math.PI * 2)
      ctx.fillStyle = DYNASTY_COLORS[cell.leader.dynasty] ?? '#999'
      ctx.fill()

      // Inner circle: leader color
      ctx.beginPath()
      ctx.arc(cx, cy, innerRadius, 0, Math.PI * 2)
      ctx.fillStyle = COLORS[cell.leader.color]
      ctx.fill()

      // Dynasty first letter
      ctx.fillStyle = '#fff'
      ctx.font = `bold ${Math.round(cellSize * 0.24)}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(cell.leader.dynasty[0].toUpperCase(), cx, cy)
    }
  }

  // 6. Draw monuments (2x2 two-tone blocks)
  const drawnMonuments = new Set<string>()
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const cell = state.board[row][col]
      if (!cell.monument || drawnMonuments.has(cell.monument)) continue
      drawnMonuments.add(cell.monument)

      const monument = state.monuments.find((m) => m.id === cell.monument)
      if (!monument || !monument.position) continue

      const mx = monument.position.col * cellSize
      const my = monument.position.row * cellSize
      const mSize = cellSize * 2

      // Color 1: top-left triangle
      ctx.beginPath()
      ctx.moveTo(mx, my)
      ctx.lineTo(mx + mSize, my)
      ctx.lineTo(mx, my + mSize)
      ctx.closePath()
      ctx.fillStyle = COLORS[monument.color1]
      ctx.fill()

      // Color 2: bottom-right triangle
      ctx.beginPath()
      ctx.moveTo(mx + mSize, my)
      ctx.lineTo(mx + mSize, my + mSize)
      ctx.lineTo(mx, my + mSize)
      ctx.closePath()
      ctx.fillStyle = COLORS[monument.color2]
      ctx.fill()

      // Monument border
      ctx.strokeStyle = '#333'
      ctx.lineWidth = 2
      ctx.strokeRect(mx + 1, my + 1, mSize - 2, mSize - 2)
    }
  }

  // 7. Draw catastrophes (gray overlay with X)
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const cell = state.board[row][col]
      if (!cell.catastrophe) continue
      const x = col * cellSize
      const y = row * cellSize

      ctx.fillStyle = COLORS.catastrophe
      ctx.globalAlpha = 0.7
      ctx.fillRect(x, y, cellSize, cellSize)
      ctx.globalAlpha = 1.0

      // Draw X
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(x + 4, y + 4)
      ctx.lineTo(x + cellSize - 4, y + cellSize - 4)
      ctx.moveTo(x + cellSize - 4, y + 4)
      ctx.lineTo(x + 4, y + cellSize - 4)
      ctx.stroke()
    }
  }

  // 8. Draw treasures (gold diamond in corner)
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      const cell = state.board[row][col]
      if (!cell.hasTreasure) continue
      const x = col * cellSize + cellSize - 10
      const y = row * cellSize + 6
      const s = 5

      ctx.fillStyle = COLORS.treasure
      ctx.beginPath()
      ctx.moveTo(x, y - s)
      ctx.lineTo(x + s, y)
      ctx.lineTo(x, y + s)
      ctx.lineTo(x - s, y)
      ctx.closePath()
      ctx.fill()
    }
  }

  // 9. Draw grid lines
  ctx.strokeStyle = COLORS.grid
  ctx.lineWidth = 0.5
  for (let row = 0; row <= BOARD_ROWS; row++) {
    ctx.beginPath()
    ctx.moveTo(0, row * cellSize)
    ctx.lineTo(width, row * cellSize)
    ctx.stroke()
  }
  for (let col = 0; col <= BOARD_COLS; col++) {
    ctx.beginPath()
    ctx.moveTo(col * cellSize, 0)
    ctx.lineTo(col * cellSize, height)
    ctx.stroke()
  }

  // 10. Draw highlights for valid placements
  if (config.highlights) {
    ctx.fillStyle = COLORS.highlight
    for (const pos of config.highlights) {
      ctx.fillRect(pos.col * cellSize, pos.row * cellSize, cellSize, cellSize)
    }
  }

  // 11. Draw hovered cell highlight
  if (config.hoveredCell) {
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.strokeRect(
      config.hoveredCell.col * cellSize + 1,
      config.hoveredCell.row * cellSize + 1,
      cellSize - 2,
      cellSize - 2,
    )
  }

  // Draw selected cell highlight
  if (config.selectedCell) {
    ctx.strokeStyle = '#ff0'
    ctx.lineWidth = 3
    ctx.strokeRect(
      config.selectedCell.col * cellSize + 1,
      config.selectedCell.row * cellSize + 1,
      cellSize - 2,
      cellSize - 2,
    )
  }
}
