import { GameState, MonumentId } from '../engine/types'

interface MonumentDialogProps {
  state: GameState
  onBuildMonument: (monumentId: MonumentId) => void
  onDeclineMonument: () => void
}

const COLOR_HEX: Record<string, string> = {
  red: '#e74c3c',
  blue: '#3498db',
  green: '#2ecc71',
  black: '#888',
}

const COLOR_LABELS: Record<string, string> = {
  red: 'Temple',
  blue: 'Farm',
  green: 'Market',
  black: 'Settlement',
}

export function MonumentDialog({ state, onBuildMonument, onDeclineMonument }: MonumentDialogProps) {
  const pending = state.pendingMonument
  if (!pending) return null

  const availableMonuments = state.monuments.filter(m =>
    m.position === null && (m.color1 === pending.color || m.color2 === pending.color)
  )

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
          Build a Monument?
        </div>

        <div style={{ fontSize: '13px', color: '#888', marginBottom: '16px' }}>
          A 2x2 {COLOR_LABELS[pending.color]} region at ({pending.position.row}, {pending.position.col})
        </div>

        {availableMonuments.length === 0 ? (
          <div style={{ color: '#666', fontSize: '13px', marginBottom: '12px' }}>
            No monuments available for this color.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
            {availableMonuments.map(monument => (
              <button
                key={monument.id}
                onClick={() => onBuildMonument(monument.id)}
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
                <div style={{ display: 'flex', gap: '3px' }}>
                  <div style={{
                    width: '20px',
                    height: '20px',
                    borderRadius: '3px',
                    background: COLOR_HEX[monument.color1],
                  }} />
                  <div style={{
                    width: '20px',
                    height: '20px',
                    borderRadius: '3px',
                    background: COLOR_HEX[monument.color2],
                  }} />
                </div>
                <span>
                  Build {COLOR_LABELS[monument.color1]}/{COLOR_LABELS[monument.color2]} Monument
                </span>
              </button>
            ))}
          </div>
        )}

        <button
          onClick={onDeclineMonument}
          style={{
            width: '100%',
            padding: '10px',
            borderRadius: '6px',
            border: '1px solid #333',
            background: '#1a1a2e',
            color: '#888',
            cursor: 'pointer',
            fontSize: '13px',
            fontFamily: 'sans-serif',
          }}
        >
          Decline
        </button>
      </div>
    </div>
  )
}
