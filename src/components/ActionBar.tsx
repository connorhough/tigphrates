import { GameState, LeaderColor, MonumentId } from '../engine/types'

interface ActionBarProps {
  state: GameState
  selectedLeader: LeaderColor | null
  onSelectLeader: (color: LeaderColor | null) => void
  onSwapTiles: () => void
  onPass: () => void
  onCommitSupport: () => void
  onBuildMonument: (monumentId: MonumentId) => void
  onDeclineMonument: () => void
  onWithdrawLeader: (color: LeaderColor) => void
  disabled?: boolean
}

const LEADER_DISPLAY: Record<LeaderColor, { label: string; color: string }> = {
  red: { label: 'Priest', color: '#e74c3c' },
  blue: { label: 'Farmer', color: '#3498db' },
  green: { label: 'Trader', color: '#2ecc71' },
  black: { label: 'King', color: '#555' },
}

const MONUMENT_COLORS: Record<string, string> = {
  red: '#e74c3c',
  blue: '#3498db',
  green: '#2ecc71',
  black: '#555',
}

const btnBase: React.CSSProperties = {
  padding: '6px 14px',
  borderRadius: '4px',
  border: '1px solid #0f3460',
  background: '#0f1b3e',
  color: '#ccc',
  cursor: 'pointer',
  fontSize: '13px',
  fontFamily: 'sans-serif',
}

const btnActive: React.CSSProperties = {
  ...btnBase,
  border: '2px solid #fff',
  background: '#1a2a5e',
  color: '#fff',
}

export function ActionBar({
  state,
  selectedLeader,
  onSelectLeader,
  onSwapTiles,
  onPass,
  onCommitSupport,
  onBuildMonument,
  onDeclineMonument,
  onWithdrawLeader,
  disabled,
}: ActionBarProps) {
  const currentPlayer = state.players[state.currentPlayer]
  const phase = state.turnPhase

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '8px 16px',
      background: '#16213e',
      borderTop: '1px solid #0f3460',
      fontFamily: 'sans-serif',
      fontSize: '13px',
      color: '#e0e0e0',
      flexWrap: 'wrap',
      minHeight: '44px',
    }}>
      {/* Action phase controls */}
      {phase === 'action' && (
        <>
          <span style={{ color: '#888', marginRight: '4px' }}>Leaders:</span>
          {currentPlayer.leaders
            .filter(l => l.position === null)
            .map(leader => {
              const display = LEADER_DISPLAY[leader.color]
              const isSelected = selectedLeader === leader.color
              return (
                <button
                  key={leader.color}
                  disabled={disabled}
                  style={isSelected ? btnActive : btnBase}
                  onClick={() => onSelectLeader(isSelected ? null : leader.color)}
                >
                  <span style={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: display.color,
                    marginRight: '4px',
                  }} />
                  {display.label}
                </button>
              )
            })}

          {/* Withdraw on-board leaders */}
          {currentPlayer.leaders
            .filter(l => l.position !== null)
            .map(leader => {
              const display = LEADER_DISPLAY[leader.color]
              return (
                <button
                  key={`withdraw-${leader.color}`}
                  disabled={disabled}
                  style={{ ...btnBase, opacity: 0.7 }}
                  onClick={() => onWithdrawLeader(leader.color)}
                  title={`Withdraw ${display.label}`}
                >
                  <span style={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: display.color,
                    marginRight: '4px',
                  }} />
                  Withdraw
                </button>
              )
            })}

          <div style={{ width: '1px', height: '24px', background: '#0f3460', margin: '0 4px' }} />

          <button
            disabled={disabled || currentPlayer.hand.length === 0}
            style={{
              ...btnBase,
              opacity: (disabled || currentPlayer.hand.length === 0) ? 0.4 : 1,
            }}
            onClick={onSwapTiles}
          >
            Swap Tiles
          </button>

          <button
            disabled={disabled}
            style={btnBase}
            onClick={onPass}
          >
            Pass
          </button>
        </>
      )}

      {/* Conflict support phase */}
      {phase === 'conflictSupport' && (
        <>
          <span style={{ color: '#f39c12' }}>
            Conflict! Select tiles from hand to commit, then:
          </span>
          <button
            disabled={disabled}
            style={btnBase}
            onClick={onCommitSupport}
          >
            Commit Support
          </button>
        </>
      )}

      {/* War order choice */}
      {phase === 'warOrderChoice' && state.pendingConflict?.pendingWarColors && (
        <>
          <span style={{ color: '#f39c12' }}>
            Choose which war to resolve:
          </span>
          {state.pendingConflict.pendingWarColors.map(color => {
            const display = LEADER_DISPLAY[color]
            return (
              <button
                key={color}
                disabled={disabled}
                style={btnBase}
                onClick={() => {
                  /* war order choice handled via dispatch in App */
                }}
              >
                <span style={{
                  display: 'inline-block',
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: display.color,
                  marginRight: '4px',
                }} />
                {display.label}
              </button>
            )
          })}
        </>
      )}

      {/* Monument choice phase */}
      {phase === 'monumentChoice' && (
        <>
          <span style={{ color: '#f39c12' }}>Build a monument?</span>
          {state.monuments
            .filter(m => m.position === null)
            .filter(m => {
              // Only show monuments matching the pending monument color
              if (!state.pendingMonument) return false
              return m.color1 === state.pendingMonument.color || m.color2 === state.pendingMonument.color
            })
            .map(monument => (
              <button
                key={monument.id}
                disabled={disabled}
                style={btnBase}
                onClick={() => onBuildMonument(monument.id)}
              >
                <span style={{
                  display: 'inline-block',
                  width: '8px',
                  height: '8px',
                  borderRadius: '2px',
                  background: MONUMENT_COLORS[monument.color1],
                  marginRight: '2px',
                }} />
                <span style={{
                  display: 'inline-block',
                  width: '8px',
                  height: '8px',
                  borderRadius: '2px',
                  background: MONUMENT_COLORS[monument.color2],
                  marginRight: '4px',
                }} />
                {monument.id}
              </button>
            ))}
          <button
            disabled={disabled}
            style={btnBase}
            onClick={onDeclineMonument}
          >
            Decline
          </button>
        </>
      )}

      {/* Game over */}
      {phase === 'gameOver' && (
        <span style={{ color: '#e74c3c', fontWeight: 'bold' }}>
          Game Over
        </span>
      )}
    </div>
  )
}
