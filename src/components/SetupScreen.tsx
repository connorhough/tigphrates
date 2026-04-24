import { useState } from 'react'

interface SetupScreenProps {
  onStartGame: (playerCount: number, aiFlags: boolean[]) => void
}

const DYNASTIES = [
  { name: 'Archer', color: 'var(--dynasty-archer)', symbol: '🏹' },
  { name: 'Bull', color: 'var(--dynasty-bull)', symbol: '🐂' },
  { name: 'Pot', color: 'var(--dynasty-pot)', symbol: '🏺' },
  { name: 'Lion', color: 'var(--dynasty-lion)', symbol: '🦁' },
]

export function SetupScreen({ onStartGame }: SetupScreenProps) {
  const [playerCount, setPlayerCount] = useState(2)
  const [aiFlags, setAiFlags] = useState([false, true, true, true])

  const toggleAI = (index: number) => {
    setAiFlags((prev) => {
      const next = [...prev]
      next[index] = !next[index]
      return next
    })
  }

  const handleStart = () => {
    onStartGame(playerCount, aiFlags.slice(0, playerCount))
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        padding: 'calc(20px + var(--sat)) calc(20px + var(--sal)) calc(20px + var(--sab)) calc(20px + var(--sar))',
        background: 'var(--paper)',
      }}
    >
      <div
        style={{
          background: 'var(--paper-dark)',
          padding: '28px 24px',
          width: '100%',
          maxWidth: 420,
          border: '1px solid var(--rule)',
        }}
      >
        <h1
          style={{
            margin: '0 0 4px',
            fontFamily: 'var(--font-display)',
            fontSize: 26,
            fontWeight: 700,
            color: 'var(--ink)',
            textAlign: 'center',
            letterSpacing: 0.5,
          }}
        >
          Tigris &amp; Euphrates
        </h1>
        <p
          style={{
            margin: '0 0 24px',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--ink-faint)',
            textAlign: 'center',
            textTransform: 'uppercase',
            letterSpacing: 2,
          }}
        >
          Mesopotamia · 3500 BC
        </p>

        <div style={{ marginBottom: 20 }}>
          <label
            style={{
              display: 'block',
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--ink-faint)',
              textTransform: 'uppercase',
              letterSpacing: 1.5,
              marginBottom: 8,
            }}
          >
            Players
          </label>
          <div style={{ display: 'flex', gap: 6 }}>
            {[2, 3, 4].map((n) => (
              <button
                key={n}
                onClick={() => setPlayerCount(n)}
                style={{
                  flex: 1,
                  padding: '12px',
                  minHeight: 48,
                  border:
                    playerCount === n
                      ? '1px solid var(--accent)'
                      : '1px solid var(--rule)',
                  background:
                    playerCount === n ? 'var(--paper-light)' : 'var(--paper)',
                  color:
                    playerCount === n ? 'var(--accent)' : 'var(--ink-light)',
                  fontFamily: 'var(--font-display)',
                  fontSize: 18,
                  fontWeight: 700,
                  cursor: 'pointer',
                }}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
            marginBottom: 24,
          }}
        >
          {Array.from({ length: playerCount }, (_, i) => {
            const dynasty = DYNASTIES[i]
            return (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  padding: '10px 14px',
                  minHeight: 48,
                  background: 'var(--paper)',
                  border: '1px solid var(--rule)',
                  borderLeft: `3px solid ${dynasty.color}`,
                }}
              >
                <span style={{ fontSize: 22 }}>{dynasty.symbol}</span>
                <span
                  style={{
                    flex: 1,
                    fontFamily: 'var(--font-display)',
                    fontSize: 14,
                    fontWeight: 700,
                    color: dynasty.color,
                  }}
                >
                  {dynasty.name}
                </span>
                <button
                  onClick={() => toggleAI(i)}
                  style={{
                    padding: '8px 14px',
                    minHeight: 36,
                    minWidth: 72,
                    border: '1px solid var(--rule)',
                    background: aiFlags[i]
                      ? 'var(--paper-dark)'
                      : 'var(--paper-light)',
                    color: aiFlags[i] ? 'var(--ink-faint)' : dynasty.color,
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    fontWeight: 500,
                    textTransform: 'uppercase',
                    letterSpacing: 1,
                    cursor: 'pointer',
                  }}
                >
                  {aiFlags[i] ? 'AI' : 'Human'}
                </button>
              </div>
            )
          })}
        </div>

        <button
          onClick={handleStart}
          style={{
            width: '100%',
            padding: 14,
            minHeight: 52,
            border: '1px solid var(--accent)',
            background: 'var(--paper-light)',
            color: 'var(--accent)',
            fontFamily: 'var(--font-mono)',
            fontSize: 13,
            fontWeight: 500,
            cursor: 'pointer',
            textTransform: 'uppercase',
            letterSpacing: 2,
          }}
        >
          Start Game
        </button>
      </div>
    </div>
  )
}
