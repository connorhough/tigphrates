import { GameState, Dynasty, TileColor, LeaderColor } from '../engine/types'

interface PlayerPanelProps {
  state: GameState
}

const DYNASTY_COLORS: Record<Dynasty, string> = {
  archer: '#e74c3c',
  bull: '#3498db',
  pot: '#2ecc71',
  lion: '#f39c12',
}

const TILE_DISPLAY: Record<TileColor, { label: string; color: string }> = {
  red: { label: 'Temple', color: '#e74c3c' },
  blue: { label: 'Farm', color: '#3498db' },
  green: { label: 'Market', color: '#2ecc71' },
  black: { label: 'Settlement', color: '#333' },
}

const LEADER_DISPLAY: Record<LeaderColor, { label: string; color: string }> = {
  red: { label: 'Priest', color: '#e74c3c' },
  blue: { label: 'Farmer', color: '#3498db' },
  green: { label: 'Trader', color: '#2ecc71' },
  black: { label: 'King', color: '#555' },
}

export function PlayerPanel({ state }: PlayerPanelProps) {
  return (
    <div style={{
      width: '220px',
      background: '#16213e',
      borderLeft: '2px solid #0f3460',
      padding: '12px',
      overflowY: 'auto',
      fontFamily: 'sans-serif',
      fontSize: '13px',
      color: '#e0e0e0',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px',
    }}>
      {state.players.map((player, idx) => {
        const isCurrent = idx === state.currentPlayer
        const dynastyColor = DYNASTY_COLORS[player.dynasty]

        return (
          <div key={idx} style={{
            padding: '10px',
            borderRadius: '6px',
            background: isCurrent ? '#1a2a5e' : '#0f1b3e',
            border: isCurrent ? `2px solid ${dynastyColor}` : '2px solid transparent',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              marginBottom: '8px',
            }}>
              <div style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: dynastyColor,
              }} />
              <span style={{
                fontWeight: 'bold',
                textTransform: 'capitalize',
                color: dynastyColor,
              }}>
                {player.dynasty}
              </span>
              {player.isAI && (
                <span style={{ color: '#666', fontSize: '11px' }}>(AI)</span>
              )}
            </div>

            {/* Scores */}
            <div style={{ marginBottom: '6px' }}>
              <div style={{ color: '#888', fontSize: '11px', marginBottom: '3px' }}>Score</div>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                {(Object.keys(player.score) as TileColor[]).map(color => (
                  <div key={color} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '3px',
                  }}>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '2px',
                      background: TILE_DISPLAY[color].color,
                    }} />
                    <span style={{ fontSize: '12px' }}>{player.score[color]}</span>
                  </div>
                ))}
                <div style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                  <span style={{ fontSize: '10px', color: '#f1c40f' }}>T</span>
                  <span style={{ fontSize: '12px' }}>{player.treasures}</span>
                </div>
              </div>
            </div>

            {/* Leaders */}
            <div style={{ marginBottom: '6px' }}>
              <div style={{ color: '#888', fontSize: '11px', marginBottom: '3px' }}>Leaders</div>
              <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                {player.leaders.map(leader => {
                  const display = LEADER_DISPLAY[leader.color]
                  const onBoard = leader.position !== null
                  return (
                    <div key={leader.color} style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '2px',
                      opacity: onBoard ? 1 : 0.4,
                    }} title={`${display.label}: ${onBoard ? 'on board' : 'in hand'}`}>
                      <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: display.color,
                        border: onBoard ? '1px solid #fff' : '1px solid #555',
                      }} />
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Catastrophes */}
            <div style={{ color: '#888', fontSize: '11px' }}>
              Catastrophes: {player.catastrophesRemaining}
            </div>
          </div>
        )
      })}
    </div>
  )
}
