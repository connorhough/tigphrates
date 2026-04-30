import { useCallback, useState } from 'react'
import { TileColor } from '../engine/types'
import { Overlay, Sheet, Header } from './ConflictDialog'

interface SwapDialogProps {
  hand: TileColor[]
  onConfirm: (indices: number[]) => void
  onCancel: () => void
}

const TILE_VAR: Record<TileColor, string> = {
  red: 'var(--tile-red)',
  blue: 'var(--tile-blue)',
  green: 'var(--tile-green)',
  black: 'var(--tile-black)',
}

const TILE_LABELS: Record<TileColor, string> = {
  red: 'Temple',
  blue: 'Farm',
  green: 'Market',
  black: 'Settle',
}

export function SwapDialog({ hand, onConfirm, onCancel }: SwapDialogProps) {
  const [selected, setSelected] = useState<Set<number>>(new Set())

  const toggle = useCallback((idx: number) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }, [])

  const selectAll = useCallback(() => {
    setSelected(new Set(hand.map((_, i) => i)))
  }, [hand])

  const clear = useCallback(() => {
    setSelected(new Set())
  }, [])

  const handleConfirm = useCallback(() => {
    if (selected.size === 0) return
    onConfirm(Array.from(selected))
    setSelected(new Set())
  }, [selected, onConfirm])

  const count = selected.size

  return (
    <Overlay>
      <Sheet>
        <Header>Swap Tiles</Header>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--ink-faint)',
            textTransform: 'uppercase',
            letterSpacing: 1,
            marginBottom: 14,
          }}
        >
          Pick tiles to discard. Replacements drawn from bag.
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 6,
            marginBottom: 14,
          }}
        >
          {hand.map((color, idx) => {
            const isSelected = selected.has(idx)
            return (
              <button
                key={idx}
                onClick={() => toggle(idx)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '8px 10px',
                  minHeight: 40,
                  border: isSelected ? '1px solid var(--ink)' : '1px solid var(--rule)',
                  background: isSelected ? 'var(--paper-light)' : 'var(--paper)',
                  color: isSelected ? 'var(--ink)' : 'var(--ink-light)',
                  cursor: 'pointer',
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                  textTransform: 'uppercase',
                  letterSpacing: 0.5,
                }}
              >
                <div
                  style={{
                    width: 16,
                    height: 16,
                    background: TILE_VAR[color],
                  }}
                />
                <span>{TILE_LABELS[color]}</span>
              </button>
            )
          })}
        </div>

        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          <button
            onClick={selectAll}
            style={subBtn}
          >
            All
          </button>
          <button
            onClick={clear}
            style={subBtn}
          >
            Clear
          </button>
        </div>

        <div style={{ display: 'flex', gap: 6 }}>
          <button
            onClick={onCancel}
            style={cancelBtn}
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={count === 0}
            style={{
              ...confirmBtn,
              opacity: count === 0 ? 0.4 : 1,
              cursor: count === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            Swap {count > 0 ? count : ''}
          </button>
        </div>
      </Sheet>
    </Overlay>
  )
}

const subBtn: React.CSSProperties = {
  flex: 1,
  padding: '6px 10px',
  minHeight: 32,
  border: '1px solid var(--rule)',
  background: 'transparent',
  color: 'var(--ink-faint)',
  cursor: 'pointer',
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: 1,
}

const cancelBtn: React.CSSProperties = {
  flex: 1,
  padding: 12,
  minHeight: 44,
  border: '1px solid var(--rule)',
  background: 'transparent',
  color: 'var(--ink-faint)',
  cursor: 'pointer',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  textTransform: 'uppercase',
  letterSpacing: 1,
}

const confirmBtn: React.CSSProperties = {
  flex: 2,
  padding: 12,
  minHeight: 44,
  border: '1px solid var(--ink)',
  background: 'var(--paper-light)',
  color: 'var(--ink)',
  cursor: 'pointer',
  fontFamily: 'var(--font-mono)',
  fontSize: 11,
  textTransform: 'uppercase',
  letterSpacing: 1,
  fontWeight: 600,
}
