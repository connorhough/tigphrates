import { GameState, LeaderColor } from '../engine/types'

interface WarOrderDialogProps {
  state: GameState
  onChooseWarOrder: (color: LeaderColor) => void
}

const COLOR_HEX: Record<string, string> = {
  red: '#e74c3c',
  blue: '#3498db',
  green: '#2ecc71',
  black: '#888',
}

const LEADER_LABELS: Record<string, string> = {
  red: 'Priest',
  blue: 'Farmer',
  green: 'Trader',
  black: 'King',
}

export function WarOrderDialog({ state, onChooseWarOrder }: WarOrderDialogProps) {
  const conflict = state.pendingConflict
  if (!conflict?.pendingWarColors?.length) return null

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'rgba(0,0,0,0.7)',
      zIndex: 1000,
    }}>
      <div style={{
        background: '#1a1a2e',
        border: '2px solid #0f3460',
        borderRadius: '12px',
        padding: '24px 32px',
        minWidth: '320px',
        maxWidth: '440px',
        fontFamily: 'sans-serif',
        color: '#e0e0e0',
      }}>
        <div style={{
          fontSize: '18px',
          fontWeight: 'bold',
          marginBottom: '8px',
        }}>
          Choose War to Resolve
        </div>

        <div style={{ fontSize: '13px', color: '#888', marginBottom: '16px' }}>
          Multiple wars triggered. Choose which to resolve first:
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {conflict.pendingWarColors.map(color => (
            <button
              key={color}
              onClick={() => onChooseWarOrder(color)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '12px 16px',
                borderRadius: '8px',
                border: '1px solid #0f3460',
                background: '#0f1b3e',
                color: '#e0e0e0',
                cursor: 'pointer',
                fontSize: '14px',
                fontFamily: 'sans-serif',
              }}
            >
              <div style={{
                width: '16px',
                height: '16px',
                borderRadius: '50%',
                background: COLOR_HEX[color],
              }} />
              <span>{LEADER_LABELS[color]} War</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
