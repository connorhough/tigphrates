import { GameState, TileColor, Dynasty } from '../engine/types'
import { calculateFinalScores } from '../engine/turn'
import { COLORS, DYNASTY_COLORS } from '../rendering/colors'

interface GameOverScreenProps {
  state: GameState
  onPlayAgain: () => void
}

const TILE_COLORS: TileColor[] = ['red', 'blue', 'green', 'black']

const COLOR_LABELS: Record<TileColor, string> = {
  red: 'Temple',
  blue: 'Farm',
  green: 'Market',
  black: 'Settlement',
}

const DYNASTY_LABELS: Record<Dynasty, string> = {
  archer: 'Archer',
  bull: 'Bull',
  pot: 'Pot',
  lion: 'Lion',
}

export function GameOverScreen({ state, onPlayAgain }: GameOverScreenProps) {
  const results = calculateFinalScores(state)
  const winnerScore = results[0].finalScore
  // Handle ties: all players matching the top score are winners
  const winnerIndices = new Set(
    results.filter(r => r.finalScore === winnerScore).map(r => r.playerIndex)
  )

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
    }}>
      <div style={{
        background: '#16213e',
        border: '1px solid #0f3460',
        borderRadius: '12px',
        padding: '32px',
        maxWidth: '640px',
        width: '90%',
        maxHeight: '90vh',
        overflowY: 'auto',
        fontFamily: 'sans-serif',
        color: '#e0e0e0',
      }}>
        <h1 style={{
          textAlign: 'center',
          margin: '0 0 24px 0',
          fontSize: '28px',
          color: '#f1c40f',
          letterSpacing: '1px',
        }}>
          Game Over
        </h1>

        {/* Winner announcement */}
        <div style={{
          textAlign: 'center',
          marginBottom: '24px',
          fontSize: '18px',
        }}>
          {winnerIndices.size === 1 ? (
            <span>
              <span style={{ color: DYNASTY_COLORS[state.players[results[0].playerIndex].dynasty] }}>
                {DYNASTY_LABELS[state.players[results[0].playerIndex].dynasty]}
              </span>
              {' '}wins with a score of{' '}
              <span style={{ color: '#f1c40f', fontWeight: 'bold' }}>{winnerScore}</span>!
            </span>
          ) : (
            <span>
              Tie between{' '}
              {results
                .filter(r => winnerIndices.has(r.playerIndex))
                .map((r, i, arr) => (
                  <span key={r.playerIndex}>
                    <span style={{ color: DYNASTY_COLORS[state.players[r.playerIndex].dynasty] }}>
                      {DYNASTY_LABELS[state.players[r.playerIndex].dynasty]}
                    </span>
                    {i < arr.length - 1 ? ' and ' : ''}
                  </span>
                ))}
              {' '}with a score of{' '}
              <span style={{ color: '#f1c40f', fontWeight: 'bold' }}>{winnerScore}</span>!
            </span>
          )}
        </div>

        {/* Player cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {results.map((result, rank) => {
            const player = state.players[result.playerIndex]
            const isWinner = winnerIndices.has(result.playerIndex)
            const dynastyColor = DYNASTY_COLORS[player.dynasty]

            return (
              <div key={result.playerIndex} style={{
                background: '#0f1b3e',
                border: isWinner ? '2px solid #f1c40f' : '1px solid #0f3460',
                borderRadius: '8px',
                padding: '16px',
                boxShadow: isWinner ? '0 0 12px rgba(241, 196, 15, 0.3)' : 'none',
              }}>
                {/* Player header */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '12px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '14px', color: '#888' }}>#{rank + 1}</span>
                    <span style={{
                      fontSize: '16px',
                      fontWeight: 'bold',
                      color: dynastyColor,
                    }}>
                      {DYNASTY_LABELS[player.dynasty]}
                    </span>
                    {isWinner && (
                      <span style={{ color: '#f1c40f', fontSize: '14px' }}>Winner</span>
                    )}
                  </div>
                  <div style={{
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: isWinner ? '#f1c40f' : '#e0e0e0',
                  }}>
                    {result.finalScore}
                  </div>
                </div>

                {/* Score table */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'auto repeat(4, 1fr)',
                  gap: '4px 12px',
                  fontSize: '13px',
                }}>
                  {/* Header row */}
                  <div style={{ color: '#888' }}></div>
                  {TILE_COLORS.map(color => (
                    <div key={color} style={{
                      color: COLORS[color],
                      textAlign: 'center',
                      fontWeight: 'bold',
                    }}>
                      {COLOR_LABELS[color]}
                    </div>
                  ))}

                  {/* Base scores */}
                  <div style={{ color: '#888' }}>Base</div>
                  {TILE_COLORS.map(color => (
                    <div key={color} style={{ textAlign: 'center' }}>
                      {player.score[color]}
                    </div>
                  ))}

                  {/* Treasure bonus */}
                  {player.treasures > 0 && (
                    <>
                      <div style={{ color: COLORS.treasure }}>+Treasure</div>
                      {TILE_COLORS.map(color => {
                        const bonus = result.colorScores[color] - player.score[color]
                        return (
                          <div key={color} style={{
                            textAlign: 'center',
                            color: bonus > 0 ? COLORS.treasure : '#555',
                          }}>
                            {bonus > 0 ? `+${bonus}` : '-'}
                          </div>
                        )
                      })}
                    </>
                  )}

                  {/* Final scores */}
                  <div style={{ color: '#ccc', fontWeight: 'bold' }}>Final</div>
                  {TILE_COLORS.map(color => (
                    <div key={color} style={{
                      textAlign: 'center',
                      fontWeight: 'bold',
                      color: result.colorScores[color] === result.finalScore ? '#f1c40f' : '#e0e0e0',
                    }}>
                      {result.colorScores[color]}
                    </div>
                  ))}
                </div>

                {/* Treasures note */}
                {player.treasures > 0 && (
                  <div style={{
                    marginTop: '8px',
                    fontSize: '12px',
                    color: '#888',
                  }}>
                    {player.treasures} treasure{player.treasures !== 1 ? 's' : ''} assigned optimally
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Play Again button */}
        <div style={{ textAlign: 'center', marginTop: '24px' }}>
          <button
            onClick={onPlayAgain}
            style={{
              padding: '12px 32px',
              borderRadius: '8px',
              border: '2px solid #f1c40f',
              background: 'transparent',
              color: '#f1c40f',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold',
              fontFamily: 'sans-serif',
              letterSpacing: '1px',
              transition: 'background 0.2s',
            }}
            onMouseEnter={e => {
              ;(e.target as HTMLButtonElement).style.background = 'rgba(241, 196, 15, 0.15)'
            }}
            onMouseLeave={e => {
              ;(e.target as HTMLButtonElement).style.background = 'transparent'
            }}
          >
            Play Again
          </button>
        </div>
      </div>
    </div>
  )
}
