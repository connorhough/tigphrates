import { GameState, Dynasty, TileColor, LeaderColor } from '../engine/types'

interface PlayerPanelProps {
  state: GameState
}

const DYNASTY_VAR: Record<Dynasty, string> = {
  archer: 'var(--dynasty-archer)',
  bull: 'var(--dynasty-bull)',
  pot: 'var(--dynasty-pot)',
  lion: 'var(--dynasty-lion)',
}

const TILE_VAR: Record<TileColor, string> = {
  red: 'var(--tile-red)',
  blue: 'var(--tile-blue)',
  green: 'var(--tile-green)',
  black: 'var(--tile-black)',
}

const LEADER_VAR: Record<LeaderColor, string> = TILE_VAR

export function PlayerPanel({ state }: PlayerPanelProps) {
  return (
    <div
      style={{
        width: 180,
        flexShrink: 0,
        background: 'var(--paper-dark)',
        borderLeft: '1px solid var(--rule)',
        padding: 8,
        paddingRight: 'calc(8px + var(--sar))',
        overflowY: 'auto',
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: 'var(--ink-light)',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}
    >
      {state.players.map((player, idx) => {
        const isCurrent = idx === state.currentPlayer
        const dynastyColor = DYNASTY_VAR[player.dynasty]

        return (
          <div
            key={idx}
            style={{
              padding: 8,
              background: isCurrent ? 'var(--paper-light)' : 'var(--paper)',
              borderLeft: `2px solid ${isCurrent ? dynastyColor : 'transparent'}`,
              border: '1px solid var(--rule-light)',
              borderLeftWidth: 2,
              borderLeftColor: isCurrent ? dynastyColor : 'var(--rule-light)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                marginBottom: 6,
              }}
            >
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: dynastyColor,
                }}
              />
              <span
                style={{
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: 1,
                  color: dynastyColor,
                  fontSize: 10,
                }}
              >
                {player.dynasty}
              </span>
              {player.isAI && (
                <span style={{ color: 'var(--ink-faint)', fontSize: 9 }}>AI</span>
              )}
            </div>

            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
              {(Object.keys(player.score) as TileColor[]).map((color) => (
                <div
                  key={color}
                  style={{ display: 'flex', alignItems: 'center', gap: 3 }}
                >
                  <div
                    style={{
                      width: 6,
                      height: 6,
                      background: TILE_VAR[color],
                    }}
                  />
                  <span style={{ color: 'var(--ink)', fontSize: 11 }}>
                    {player.score[color]}
                  </span>
                </div>
              ))}
              <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <span style={{ fontSize: 9, color: 'var(--treasure)' }}>T</span>
                <span style={{ color: 'var(--ink)', fontSize: 11 }}>
                  {player.treasures}
                </span>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
              {player.leaders.map((leader) => {
                const onBoard = leader.position !== null
                return (
                  <div
                    key={leader.color}
                    title={`${leader.color}: ${onBoard ? 'on board' : 'in hand'}`}
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      background: LEADER_VAR[leader.color],
                      opacity: onBoard ? 1 : 0.3,
                      border: onBoard
                        ? '1px solid var(--ink)'
                        : '1px solid var(--rule)',
                    }}
                  />
                )
              })}
            </div>

            {player.catastrophesRemaining > 0 && (
              <div
                style={{
                  marginTop: 4,
                  color: 'var(--ink-faint)',
                  fontSize: 9,
                  letterSpacing: 0.5,
                  textTransform: 'uppercase',
                }}
              >
                Cat ×{player.catastrophesRemaining}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
