import { TileColor } from '../engine/types'

interface HandPanelProps {
  hand: TileColor[]
  selectedTile: TileColor | null
  onSelectTile: (color: TileColor | null) => void
  disabled?: boolean
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

export function HandPanel({ hand, selectedTile, onSelectTile, disabled }: HandPanelProps) {
  const counts: Partial<Record<TileColor, number>> = {}
  for (const tile of hand) counts[tile] = (counts[tile] ?? 0) + 1
  const colors = Object.keys(counts) as TileColor[]

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 10px',
        paddingLeft: 'calc(10px + var(--sal))',
        paddingRight: 'calc(10px + var(--sar))',
        background: 'var(--paper-dark)',
        borderTop: '1px solid var(--rule)',
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: 'var(--ink-light)',
        minHeight: 48,
        overflowX: 'auto',
        flexShrink: 0,
        scrollSnapType: 'x mandatory',
      }}
    >
      <span
        style={{
          color: 'var(--ink-faint)',
          textTransform: 'uppercase',
          letterSpacing: 1,
          fontSize: 9,
          flexShrink: 0,
        }}
      >
        Hand
      </span>
      {colors.length === 0 && (
        <span style={{ color: 'var(--ink-faint)', fontStyle: 'italic' }}>empty</span>
      )}
      {colors.map((color) => {
        const isSelected = selectedTile === color
        return (
          <button
            key={color}
            disabled={disabled}
            onClick={() => onSelectTile(isSelected ? null : color)}
            style={{
              scrollSnapAlign: 'start',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 10px',
              minHeight: 36,
              borderRadius: 0,
              border: isSelected
                ? '1px solid var(--ink)'
                : '1px solid var(--rule)',
              background: isSelected ? 'var(--paper-light)' : 'var(--paper)',
              color: isSelected ? 'var(--ink)' : 'var(--ink-light)',
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.4 : 1,
              fontSize: 11,
              fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase',
              letterSpacing: 0.5,
              flexShrink: 0,
            }}
          >
            <div
              style={{
                width: 14,
                height: 14,
                background: TILE_VAR[color],
              }}
            />
            <span>{TILE_LABELS[color]}</span>
            <span
              style={{
                background: 'var(--rule)',
                color: 'var(--ink)',
                padding: '1px 6px',
                fontSize: 10,
              }}
            >
              {counts[color]}
            </span>
          </button>
        )
      })}
    </div>
  )
}
