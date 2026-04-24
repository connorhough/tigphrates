import { useRef, useEffect, useState, useCallback, useLayoutEffect } from 'react'
import { drawBoard, RenderConfig } from '../rendering/boardRenderer'
import { GameState, Position, BOARD_ROWS, BOARD_COLS } from '../engine/types'

interface GameBoardProps {
  state: GameState
  highlights?: Position[]
  onCellClick?: (pos: Position) => void
}

export function GameBoard({ state, highlights, onCellClick }: GameBoardProps) {
  const wrapperRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [cellSize, setCellSize] = useState(36)
  const [hoveredCell, setHoveredCell] = useState<Position | null>(null)

  // Fit cellSize to wrapper. Board is BOARD_COLS wide × BOARD_ROWS tall.
  useLayoutEffect(() => {
    const wrapper = wrapperRef.current
    if (!wrapper) return
    const update = () => {
      const rect = wrapper.getBoundingClientRect()
      const byW = Math.floor(rect.width / BOARD_COLS)
      const byH = Math.floor(rect.height / BOARD_ROWS)
      const next = Math.max(16, Math.min(byW, byH))
      setCellSize(next)
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(wrapper)
    return () => ro.disconnect()
  }, [])

  // Render w/ DPR scaling
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const cssW = cellSize * BOARD_COLS
    const cssH = cellSize * BOARD_ROWS
    canvas.width = Math.floor(cssW * dpr)
    canvas.height = Math.floor(cssH * dpr)
    canvas.style.width = `${cssW}px`
    canvas.style.height = `${cssH}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const config: RenderConfig = { cellSize, highlights, hoveredCell }
    drawBoard(ctx, state, config)
  }, [state, highlights, hoveredCell, cellSize])

  const pointerToCell = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>): Position => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      return { row: Math.floor(y / cellSize), col: Math.floor(x / cellSize) }
    },
    [cellSize],
  )

  const isCoarse =
    typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (isCoarse) return
      const pos = pointerToCell(e)
      if (pos.row >= 0 && pos.row < BOARD_ROWS && pos.col >= 0 && pos.col < BOARD_COLS) {
        setHoveredCell((prev) =>
          prev && prev.row === pos.row && prev.col === pos.col ? prev : pos,
        )
      } else {
        setHoveredCell(null)
      }
    },
    [pointerToCell, isCoarse],
  )

  const handlePointerLeave = useCallback(() => setHoveredCell(null), [])

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!onCellClick) return
      const pos = pointerToCell(e)
      if (pos.row >= 0 && pos.row < BOARD_ROWS && pos.col >= 0 && pos.col < BOARD_COLS) {
        onCellClick(pos)
      }
    },
    [onCellClick, pointerToCell],
  )

  return (
    <div
      ref={wrapperRef}
      style={{
        flex: 1,
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        padding: 4,
      }}
    >
      <canvas
        ref={canvasRef}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onPointerUp={handlePointerUp}
        style={{ cursor: onCellClick ? 'pointer' : 'default', display: 'block' }}
      />
    </div>
  )
}
