import { GameState, MonumentId } from '../engine/types'
import { Overlay, Sheet, Header } from './ConflictDialog'

interface MonumentDialogProps {
  state: GameState
  onBuildMonument: (monumentId: MonumentId) => void
  onDeclineMonument: () => void
}

const TILE_VAR: Record<string, string> = {
  red: 'var(--tile-red)',
  blue: 'var(--tile-blue)',
  green: 'var(--tile-green)',
  black: 'var(--tile-black)',
}

const LABELS: Record<string, string> = {
  red: 'Temple',
  blue: 'Farm',
  green: 'Market',
  black: 'Settlement',
}

export function MonumentDialog({
  state,
  onBuildMonument,
  onDeclineMonument,
}: MonumentDialogProps) {
  const pending = state.pendingMonument
  if (!pending) return null

  const available = state.monuments.filter(
    (m) =>
      m.position === null && (m.color1 === pending.color || m.color2 === pending.color),
  )

  return (
    <Overlay>
      <Sheet>
        <Header>Build a Monument?</Header>
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
          2×2 {LABELS[pending.color]} at ({pending.position.row}, {pending.position.col})
        </div>

        {available.length === 0 ? (
          <div style={{ color: 'var(--ink-faint)', marginBottom: 12 }}>
            No monuments available for this color.
          </div>
        ) : (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
              marginBottom: 14,
            }}
          >
            {available.map((m) => (
              <button
                key={m.id}
                onClick={() => onBuildMonument(m.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: 12,
                  minHeight: 48,
                  border: '1px solid var(--rule)',
                  background: 'var(--paper-light)',
                  color: 'var(--ink)',
                  cursor: 'pointer',
                  fontFamily: 'var(--font-body)',
                  fontSize: 14,
                  textAlign: 'left',
                }}
              >
                <div style={{ display: 'flex', gap: 3 }}>
                  <div
                    style={{
                      width: 22,
                      height: 22,
                      background: TILE_VAR[m.color1],
                    }}
                  />
                  <div
                    style={{
                      width: 22,
                      height: 22,
                      background: TILE_VAR[m.color2],
                    }}
                  />
                </div>
                <span>
                  Build {LABELS[m.color1]}/{LABELS[m.color2]}
                </span>
              </button>
            ))}
          </div>
        )}

        <button
          onClick={onDeclineMonument}
          style={{
            width: '100%',
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
          }}
        >
          Decline
        </button>
      </Sheet>
    </Overlay>
  )
}
