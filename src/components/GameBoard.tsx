import { useRef, useEffect, useState, useCallback } from 'react'
import { drawBoard, RenderConfig } from '../rendering/boardRenderer'
import { GameState, Position, BOARD_ROWS, BOARD_COLS } from '../engine/types'

interface GameBoardProps {
  state: GameState
  highlights?: Position[]
  onCellClick?: (pos: Position) => void
  cellSize?: number
}

export function GameBoard({ state, highlights, onCellClick, cellSize = 48 }: GameBoardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredCell, setHoveredCell] = useState<Position | null>(null)

  const width = cellSize * BOARD_COLS
  const height = cellSize * BOARD_ROWS

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const config: RenderConfig = {
      cellSize,
      highlights,
      hoveredCell,
    }

    drawBoard(ctx, state, config)
  }, [state, highlights, hoveredCell, cellSize])

  const pixelToCell = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): Position => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      return {
        row: Math.floor(y / cellSize),
        col: Math.floor(x / cellSize),
      }
    },
    [cellSize],
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = pixelToCell(e)
      if (
        pos.row >= 0 && pos.row < BOARD_ROWS &&
        pos.col >= 0 && pos.col < BOARD_COLS
      ) {
        setHoveredCell((prev) =>
          prev && prev.row === pos.row && prev.col === pos.col ? prev : pos,
        )
      } else {
        setHoveredCell(null)
      }
    },
    [pixelToCell],
  )

  const handleMouseLeave = useCallback(() => {
    setHoveredCell(null)
  }, [])

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onCellClick) return
      const pos = pixelToCell(e)
      if (
        pos.row >= 0 && pos.row < BOARD_ROWS &&
        pos.col >= 0 && pos.col < BOARD_COLS
      ) {
        onCellClick(pos)
      }
    },
    [onCellClick, pixelToCell],
  )

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      style={{ cursor: onCellClick ? 'pointer' : 'default' }}
    />
  )
}
