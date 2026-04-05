import { TileColor } from '../engine/types'

interface HandPanelProps {
  hand: TileColor[]
  selectedTile: TileColor | null
  onSelectTile: (color: TileColor | null) => void
  disabled?: boolean
}

const TILE_COLORS: Record<TileColor, string> = {
  red: '#e74c3c',
  blue: '#3498db',
  green: '#2ecc71',
  black: '#555',
}

const TILE_LABELS: Record<TileColor, string> = {
  red: 'Temple',
  blue: 'Farm',
  green: 'Market',
  black: 'Settlement',
}

export function HandPanel({ hand, selectedTile, onSelectTile, disabled }: HandPanelProps) {
  // Group tiles by color
  const counts: Partial<Record<TileColor, number>> = {}
  for (const tile of hand) {
    counts[tile] = (counts[tile] ?? 0) + 1
  }

  const colors = Object.keys(counts) as TileColor[]

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '8px 16px',
      background: '#16213e',
      borderTop: '2px solid #0f3460',
      fontFamily: 'sans-serif',
      fontSize: '13px',
      color: '#e0e0e0',
      minHeight: '50px',
    }}>
      <span style={{ color: '#888', marginRight: '8px' }}>Hand:</span>
      {colors.length === 0 && (
        <span style={{ color: '#555' }}>Empty</span>
      )}
      {colors.map(color => {
        const isSelected = selectedTile === color
        return (
          <button
            key={color}
            disabled={disabled}
            onClick={() => onSelectTile(isSelected ? null : color)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: '4px',
              border: isSelected ? '2px solid #fff' : '2px solid transparent',
              background: isSelected ? TILE_COLORS[color] : '#0f1b3e',
              color: isSelected ? '#fff' : '#ccc',
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.5 : 1,
              fontSize: '13px',
              fontFamily: 'inherit',
            }}
          >
            <div style={{
              width: '14px',
              height: '14px',
              borderRadius: '3px',
              background: TILE_COLORS[color],
            }} />
            <span>{TILE_LABELS[color]}</span>
            <span style={{
              background: 'rgba(255,255,255,0.15)',
              borderRadius: '8px',
              padding: '1px 6px',
              fontSize: '11px',
            }}>
              {counts[color]}
            </span>
          </button>
        )
      })}
    </div>
  )
}
