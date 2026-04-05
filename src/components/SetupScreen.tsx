import { useState } from 'react'

interface SetupScreenProps {
  onStartGame: (playerCount: number, aiFlags: boolean[]) => void
}

const DYNASTIES = [
  { name: 'Archer', color: '#e74c3c', symbol: '🏹' },
  { name: 'Bull', color: '#3498db', symbol: '🐂' },
  { name: 'Pot', color: '#2ecc71', symbol: '🏺' },
  { name: 'Lion', color: '#f39c12', symbol: '🦁' },
]

export function SetupScreen({ onStartGame }: SetupScreenProps) {
  const [playerCount, setPlayerCount] = useState(2)
  const [aiFlags, setAiFlags] = useState([false, true, true, true])

  const toggleAI = (index: number) => {
    setAiFlags(prev => {
      const next = [...prev]
      next[index] = !next[index]
      return next
    })
  }

  const handleStart = () => {
    onStartGame(playerCount, aiFlags.slice(0, playerCount))
  }

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      background: '#1a1a2e',
      fontFamily: 'system-ui, sans-serif',
    }}>
      <div style={{
        background: '#16213e',
        borderRadius: '16px',
        padding: '40px 48px',
        minWidth: '400px',
        maxWidth: '480px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
        border: '1px solid #0f3460',
      }}>
        <h1 style={{
          margin: '0 0 8px',
          fontSize: '32px',
          fontWeight: 600,
          color: '#e0e0e0',
          textAlign: 'center',
          letterSpacing: '-0.5px',
        }}>
          Tigris &amp; Euphrates
        </h1>
        <p style={{
          margin: '0 0 32px',
          fontSize: '13px',
          color: '#556',
          textAlign: 'center',
        }}>
          Civilization building in ancient Mesopotamia
        </p>

        {/* Player count */}
        <div style={{ marginBottom: '28px' }}>
          <label style={{
            display: 'block',
            fontSize: '12px',
            fontWeight: 600,
            color: '#8890a4',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            marginBottom: '10px',
          }}>
            Players
          </label>
          <div style={{ display: 'flex', gap: '8px' }}>
            {[2, 3, 4].map(n => (
              <button
                key={n}
                onClick={() => setPlayerCount(n)}
                style={{
                  flex: 1,
                  padding: '10px',
                  borderRadius: '8px',
                  border: playerCount === n
                    ? '2px solid #5a7ec7'
                    : '2px solid #2a2a4a',
                  background: playerCount === n ? '#1e3a6e' : '#12132a',
                  color: playerCount === n ? '#c0d0f0' : '#556',
                  fontSize: '16px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        {/* Player slots */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '32px' }}>
          {Array.from({ length: playerCount }, (_, i) => {
            const dynasty = DYNASTIES[i]
            return (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '10px 14px',
                  borderRadius: '8px',
                  background: '#12132a',
                  border: `1px solid ${dynasty.color}33`,
                }}
              >
                <span style={{ fontSize: '20px' }}>{dynasty.symbol}</span>
                <span style={{
                  flex: 1,
                  fontSize: '14px',
                  fontWeight: 600,
                  color: dynasty.color,
                }}>
                  {dynasty.name}
                </span>
                <button
                  onClick={() => toggleAI(i)}
                  style={{
                    padding: '4px 14px',
                    borderRadius: '6px',
                    border: 'none',
                    background: aiFlags[i] ? '#2a2a4a' : dynasty.color + '30',
                    color: aiFlags[i] ? '#889' : dynasty.color,
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    minWidth: '64px',
                    transition: 'all 0.15s',
                  }}
                >
                  {aiFlags[i] ? 'AI' : 'Human'}
                </button>
              </div>
            )
          })}
        </div>

        {/* Start button */}
        <button
          onClick={handleStart}
          style={{
            width: '100%',
            padding: '14px',
            borderRadius: '10px',
            border: 'none',
            background: 'linear-gradient(135deg, #1e3a6e, #0f3460)',
            color: '#c0d0f0',
            fontSize: '16px',
            fontWeight: 600,
            cursor: 'pointer',
            letterSpacing: '0.5px',
            transition: 'all 0.15s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'linear-gradient(135deg, #2a4e8e, #1a4a7a)'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'linear-gradient(135deg, #1e3a6e, #0f3460)'
          }}
        >
          Start Game
        </button>
      </div>
    </div>
  )
}
