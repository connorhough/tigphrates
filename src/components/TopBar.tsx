import { GameState, Dynasty } from '../engine/types'

interface TopBarProps {
  state: GameState
}

const DYNASTY_COLORS: Record<Dynasty, string> = {
  archer: '#e74c3c',
  bull: '#3498db',
  pot: '#2ecc71',
  lion: '#f39c12',
}

const PHASE_LABELS: Record<string, string> = {
  action: 'Action Phase',
  conflictSupport: 'Conflict - Commit Support',
  warOrderChoice: 'Choose War to Resolve',
  monumentChoice: 'Monument Decision',
  gameOver: 'Game Over',
}

export function TopBar({ state }: TopBarProps) {
  const currentPlayer = state.players[state.currentPlayer]
  const dynasty = currentPlayer.dynasty
  const dynastyColor = DYNASTY_COLORS[dynasty]

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '8px 16px',
      background: '#16213e',
      borderBottom: '2px solid #0f3460',
      color: '#e0e0e0',
      fontFamily: 'sans-serif',
      fontSize: '14px',
      minHeight: '40px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div style={{
          width: '12px',
          height: '12px',
          borderRadius: '50%',
          background: dynastyColor,
          boxShadow: `0 0 6px ${dynastyColor}`,
        }} />
        <span style={{ fontWeight: 'bold', textTransform: 'capitalize' }}>
          {dynasty}
        </span>
        <span style={{ color: '#888' }}>
          (Player {state.currentPlayer + 1})
        </span>
      </div>

      <div style={{ color: '#aaa' }}>
        {PHASE_LABELS[state.turnPhase] ?? state.turnPhase}
      </div>

      <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
        {state.turnPhase === 'action' && (
          <span>
            Actions: <strong>{state.actionsRemaining}</strong> / 2
          </span>
        )}
        <span style={{ color: '#666' }}>
          Bag: {state.bag.length}
        </span>
      </div>
    </div>
  )
}
