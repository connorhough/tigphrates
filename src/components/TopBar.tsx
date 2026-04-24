import { GameState, Dynasty } from '../engine/types'

interface TopBarProps {
  state: GameState
}

const DYNASTY_VAR: Record<Dynasty, string> = {
  archer: 'var(--dynasty-archer)',
  bull: 'var(--dynasty-bull)',
  pot: 'var(--dynasty-pot)',
  lion: 'var(--dynasty-lion)',
}

const PHASE_LABELS: Record<string, string> = {
  action: 'Action',
  conflictSupport: 'Conflict — Commit',
  warOrderChoice: 'Choose War',
  monumentChoice: 'Monument',
  gameOver: 'Game Over',
}

export function TopBar({ state }: TopBarProps) {
  const currentPlayer = state.players[state.currentPlayer]
  const dynasty = currentPlayer.dynasty
  const dynastyColor = DYNASTY_VAR[dynasty]

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 10,
        padding: '6px 12px',
        paddingTop: 'calc(6px + var(--sat))',
        paddingLeft: 'calc(12px + var(--sal))',
        paddingRight: 'calc(12px + var(--sar))',
        background: 'var(--paper-dark)',
        borderBottom: '1px solid var(--rule)',
        color: 'var(--ink)',
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        letterSpacing: 0.5,
        textTransform: 'uppercase',
        minHeight: 36,
        flexShrink: 0,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: dynastyColor,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontWeight: 700,
            color: dynastyColor,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {dynasty}
          {currentPlayer.isAI ? ' · AI' : ''}
        </span>
      </div>

      <div
        style={{
          color: 'var(--ink-light)',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {PHASE_LABELS[state.turnPhase] ?? state.turnPhase}
        {currentPlayer.isAI && state.turnPhase !== 'gameOver' && (
          <span
            style={{
              color: 'var(--dynasty-lion)',
              animation: 'pulse 1.5s ease-in-out infinite',
            }}
          >
            thinking…
          </span>
        )}
      </div>

      <div
        style={{
          display: 'flex',
          gap: 10,
          alignItems: 'center',
          color: 'var(--ink-faint)',
          flexShrink: 0,
        }}
      >
        {state.turnPhase === 'action' && (
          <span>
            <span style={{ color: 'var(--ink-light)' }}>Act</span>{' '}
            <strong style={{ color: 'var(--ink)' }}>{state.actionsRemaining}</strong>/2
          </span>
        )}
        <span>Bag {state.bag.length}</span>
      </div>
    </div>
  )
}
