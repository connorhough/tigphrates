import { GameState, LeaderColor } from '../engine/types'
import { Overlay, Sheet, Header } from './ConflictDialog'

interface WarOrderDialogProps {
  state: GameState
  onChooseWarOrder: (color: LeaderColor) => void
}

const TILE_VAR: Record<string, string> = {
  red: 'var(--tile-red)',
  blue: 'var(--tile-blue)',
  green: 'var(--tile-green)',
  black: 'var(--tile-black)',
}

const LABELS: Record<string, string> = {
  red: 'Priest',
  blue: 'Farmer',
  green: 'Trader',
  black: 'King',
}

export function WarOrderDialog({ state, onChooseWarOrder }: WarOrderDialogProps) {
  const conflict = state.pendingConflict
  if (!conflict?.pendingWarColors?.length) return null

  return (
    <Overlay>
      <Sheet>
        <Header>Choose War to Resolve</Header>
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
          Multiple wars triggered. Pick first:
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {conflict.pendingWarColors.map((color) => (
            <button
              key={color}
              onClick={() => onChooseWarOrder(color)}
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
              <div
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: '50%',
                  background: TILE_VAR[color],
                }}
              />
              <span>{LABELS[color]} War</span>
            </button>
          ))}
        </div>
      </Sheet>
    </Overlay>
  )
}
