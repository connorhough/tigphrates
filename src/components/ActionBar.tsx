import { GameState, LeaderColor } from '../engine/types'

interface ActionBarProps {
  state: GameState
  selectedLeader: LeaderColor | null
  placingCatastrophe: boolean
  onSelectLeader: (color: LeaderColor | null) => void
  onPlaceCatastrophe: (v: boolean) => void
  onSwapTiles: () => void
  onPass: () => void
  onWithdrawLeader: (color: LeaderColor) => void
  disabled?: boolean
}

const LEADER_VAR: Record<LeaderColor, { label: string; color: string }> = {
  red: { label: 'Priest', color: 'var(--tile-red)' },
  blue: { label: 'Farmer', color: 'var(--tile-blue)' },
  green: { label: 'Trader', color: 'var(--tile-green)' },
  black: { label: 'King', color: 'var(--tile-black)' },
}

const btnBase: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 5,
  padding: '6px 10px',
  minHeight: 36,
  borderRadius: 0,
  border: '1px solid var(--rule)',
  background: 'var(--paper)',
  color: 'var(--ink-light)',
  cursor: 'pointer',
  fontSize: 11,
  fontFamily: 'var(--font-mono)',
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  flexShrink: 0,
}

const btnActive: React.CSSProperties = {
  ...btnBase,
  border: '1px solid var(--ink)',
  background: 'var(--paper-light)',
  color: 'var(--ink)',
}

const dot = (color: string): React.CSSProperties => ({
  display: 'inline-block',
  width: 8,
  height: 8,
  borderRadius: '50%',
  background: color,
})

export function ActionBar({
  state,
  selectedLeader,
  placingCatastrophe,
  onSelectLeader,
  onPlaceCatastrophe,
  onSwapTiles,
  onPass,
  onWithdrawLeader,
  disabled,
}: ActionBarProps) {
  const currentPlayer = state.players[state.currentPlayer]
  const phase = state.turnPhase

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 10px',
        paddingLeft: 'calc(10px + var(--sal))',
        paddingRight: 'calc(10px + var(--sar))',
        paddingBottom: 'calc(6px + var(--sab))',
        background: 'var(--paper-dark)',
        borderTop: '1px solid var(--rule)',
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: 'var(--ink-light)',
        overflowX: 'auto',
        minHeight: 48,
        flexShrink: 0,
      }}
    >
      {phase === 'action' && (
        <>
          {currentPlayer.leaders
            .filter((l) => l.position === null)
            .map((leader) => {
              const display = LEADER_VAR[leader.color]
              const isSelected = selectedLeader === leader.color
              return (
                <button
                  key={leader.color}
                  disabled={disabled}
                  style={isSelected ? btnActive : btnBase}
                  onClick={() => onSelectLeader(isSelected ? null : leader.color)}
                >
                  <span style={dot(display.color)} />
                  {display.label}
                </button>
              )
            })}

          {currentPlayer.leaders
            .filter((l) => l.position !== null)
            .map((leader) => {
              const display = LEADER_VAR[leader.color]
              return (
                <button
                  key={`withdraw-${leader.color}`}
                  disabled={disabled}
                  style={{ ...btnBase, opacity: 0.7 }}
                  onClick={() => onWithdrawLeader(leader.color)}
                  title={`Withdraw ${display.label}`}
                >
                  <span style={dot(display.color)} />
                  Withdraw
                </button>
              )
            })}

          <div
            style={{
              width: 1,
              height: 20,
              background: 'var(--rule)',
              margin: '0 4px',
              flexShrink: 0,
            }}
          />

          <button
            disabled={disabled || currentPlayer.hand.length === 0}
            style={{
              ...btnBase,
              opacity: disabled || currentPlayer.hand.length === 0 ? 0.4 : 1,
            }}
            onClick={onSwapTiles}
          >
            Swap
          </button>

          {currentPlayer.catastrophesRemaining > 0 && (
            <button
              disabled={disabled}
              style={placingCatastrophe ? btnActive : btnBase}
              onClick={() => onPlaceCatastrophe(!placingCatastrophe)}
            >
              Cat ×{currentPlayer.catastrophesRemaining}
            </button>
          )}

          <button disabled={disabled} style={btnBase} onClick={onPass}>
            Pass
          </button>
        </>
      )}

      {phase === 'conflictSupport' && (
        <span style={{ color: 'var(--dynasty-lion)' }}>Conflict in progress…</span>
      )}
      {phase === 'warOrderChoice' && (
        <span style={{ color: 'var(--dynasty-lion)' }}>Choose which war to resolve…</span>
      )}
      {phase === 'monumentChoice' && (
        <span style={{ color: 'var(--dynasty-lion)' }}>Monument choice pending…</span>
      )}
      {phase === 'gameOver' && (
        <span style={{ color: 'var(--tile-red)', fontWeight: 700 }}>Game Over</span>
      )}
    </div>
  )
}
