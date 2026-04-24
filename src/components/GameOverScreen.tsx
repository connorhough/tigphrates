import { GameState, TileColor, Dynasty } from '../engine/types'
import { calculateFinalScores } from '../engine/turn'
import { Overlay, Sheet } from './ConflictDialog'

interface GameOverScreenProps {
  state: GameState
  onPlayAgain: () => void
}

const TILE_COLORS: TileColor[] = ['red', 'blue', 'green', 'black']

const TILE_VAR: Record<TileColor, string> = {
  red: 'var(--tile-red)',
  blue: 'var(--tile-blue)',
  green: 'var(--tile-green)',
  black: 'var(--tile-black)',
}

const LABELS: Record<TileColor, string> = {
  red: 'Temple',
  blue: 'Farm',
  green: 'Market',
  black: 'Settle',
}

const DYNASTY_VAR: Record<Dynasty, string> = {
  archer: 'var(--dynasty-archer)',
  bull: 'var(--dynasty-bull)',
  pot: 'var(--dynasty-pot)',
  lion: 'var(--dynasty-lion)',
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
  const winnerIndices = new Set(
    results.filter((r) => r.finalScore === winnerScore).map((r) => r.playerIndex),
  )

  return (
    <Overlay>
      <Sheet>
        <h1
          style={{
            textAlign: 'center',
            margin: '0 0 16px',
            fontFamily: 'var(--font-display)',
            fontSize: 26,
            fontWeight: 700,
            color: 'var(--treasure)',
            letterSpacing: 1,
          }}
        >
          Game Over
        </h1>

        <div
          style={{
            textAlign: 'center',
            marginBottom: 20,
            fontFamily: 'var(--font-body)',
            fontSize: 16,
            color: 'var(--ink-light)',
          }}
        >
          {winnerIndices.size === 1 ? (
            <>
              <span
                style={{
                  color: DYNASTY_VAR[state.players[results[0].playerIndex].dynasty],
                  fontWeight: 700,
                }}
              >
                {DYNASTY_LABELS[state.players[results[0].playerIndex].dynasty]}
              </span>{' '}
              wins —{' '}
              <span style={{ color: 'var(--treasure)', fontWeight: 700 }}>
                {winnerScore}
              </span>
            </>
          ) : (
            <>
              Tie —{' '}
              <span style={{ color: 'var(--treasure)', fontWeight: 700 }}>
                {winnerScore}
              </span>
            </>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {results.map((result, rank) => {
            const player = state.players[result.playerIndex]
            const isWinner = winnerIndices.has(result.playerIndex)
            const dynastyColor = DYNASTY_VAR[player.dynasty]

            return (
              <div
                key={result.playerIndex}
                style={{
                  background: 'var(--paper-light)',
                  border: `1px solid ${isWinner ? 'var(--treasure)' : 'var(--rule)'}`,
                  padding: 12,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: 10,
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                    }}
                  >
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 10,
                        color: 'var(--ink-faint)',
                      }}
                    >
                      #{rank + 1}
                    </span>
                    <span
                      style={{
                        fontFamily: 'var(--font-display)',
                        fontSize: 15,
                        fontWeight: 700,
                        color: dynastyColor,
                      }}
                    >
                      {DYNASTY_LABELS[player.dynasty]}
                    </span>
                    {isWinner && (
                      <span
                        style={{
                          color: 'var(--treasure)',
                          fontFamily: 'var(--font-mono)',
                          fontSize: 10,
                          textTransform: 'uppercase',
                          letterSpacing: 1,
                        }}
                      >
                        Winner
                      </span>
                    )}
                  </div>
                  <div
                    style={{
                      fontFamily: 'var(--font-display)',
                      fontSize: 22,
                      fontWeight: 700,
                      color: isWinner ? 'var(--treasure)' : 'var(--ink)',
                    }}
                  >
                    {result.finalScore}
                  </div>
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'auto repeat(4, 1fr)',
                    gap: '4px 10px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                  }}
                >
                  <div style={{ color: 'var(--ink-faint)' }}></div>
                  {TILE_COLORS.map((c) => (
                    <div
                      key={c}
                      style={{
                        color: TILE_VAR[c],
                        textAlign: 'center',
                        fontWeight: 500,
                      }}
                    >
                      {LABELS[c]}
                    </div>
                  ))}

                  <div style={{ color: 'var(--ink-faint)' }}>Base</div>
                  {TILE_COLORS.map((c) => (
                    <div
                      key={c}
                      style={{ textAlign: 'center', color: 'var(--ink-light)' }}
                    >
                      {player.score[c]}
                    </div>
                  ))}

                  {player.treasures > 0 && (
                    <>
                      <div style={{ color: 'var(--treasure)' }}>+T</div>
                      {TILE_COLORS.map((c) => {
                        const bonus = result.colorScores[c] - player.score[c]
                        return (
                          <div
                            key={c}
                            style={{
                              textAlign: 'center',
                              color:
                                bonus > 0
                                  ? 'var(--treasure)'
                                  : 'var(--ink-faint)',
                            }}
                          >
                            {bonus > 0 ? `+${bonus}` : '-'}
                          </div>
                        )
                      })}
                    </>
                  )}

                  <div style={{ color: 'var(--ink)', fontWeight: 700 }}>Fin</div>
                  {TILE_COLORS.map((c) => (
                    <div
                      key={c}
                      style={{
                        textAlign: 'center',
                        fontWeight: 700,
                        color:
                          result.colorScores[c] === result.finalScore
                            ? 'var(--treasure)'
                            : 'var(--ink)',
                      }}
                    >
                      {result.colorScores[c]}
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        <div style={{ textAlign: 'center', marginTop: 20 }}>
          <button
            onClick={onPlayAgain}
            style={{
              padding: '14px 32px',
              minHeight: 52,
              width: '100%',
              border: '1px solid var(--treasure)',
              background: 'transparent',
              color: 'var(--treasure)',
              cursor: 'pointer',
              fontFamily: 'var(--font-mono)',
              fontSize: 13,
              fontWeight: 500,
              textTransform: 'uppercase',
              letterSpacing: 2,
            }}
          >
            Play Again
          </button>
        </div>
      </Sheet>
    </Overlay>
  )
}
