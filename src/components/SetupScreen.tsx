import { useEffect, useState } from 'react'
import { SeatPolicy, DEFAULT_HEURISTIC, DEFAULT_HUMAN } from '../types/seatPolicy'
import { fetchLeaderboard, LabMember, LAB_BASE_URL } from '../lab/labApi'

interface SetupScreenProps {
  onStartGame: (seatPolicies: SeatPolicy[]) => void
  onOpenLab: () => void
  initialSeatPolicies?: SeatPolicy[]
}

const DYNASTIES = [
  { name: 'Archer', color: 'var(--dynasty-archer)', symbol: '🏹' },
  { name: 'Bull', color: 'var(--dynasty-bull)', symbol: '🐂' },
  { name: 'Pot', color: 'var(--dynasty-pot)', symbol: '🏺' },
  { name: 'Lion', color: 'var(--dynasty-lion)', symbol: '🦁' },
]

type AIKindKey = 'heuristic' | 'server' | 'onnx-default' | string  // string = pool member name

function policyForKind(kind: AIKindKey): SeatPolicy {
  if (kind === 'heuristic') return { kind: 'heuristic' }
  if (kind === 'server') return { kind: 'server' }
  if (kind === 'onnx-default') return { kind: 'onnx', modelUrl: '/policy.onnx' }
  // Pool member: lab server serves it.
  return { kind: 'onnx', modelUrl: `${LAB_BASE_URL}/pool/${kind}.onnx` }
}

function kindForPolicy(p: SeatPolicy): AIKindKey {
  if (p.kind === 'heuristic') return 'heuristic'
  if (p.kind === 'server') return 'server'
  if (p.kind === 'onnx') {
    if (p.modelUrl === '/policy.onnx') return 'onnx-default'
    const m = p.modelUrl.match(/\/pool\/([^/]+)\.onnx$/)
    if (m) return m[1]
    return 'onnx-default'
  }
  return 'heuristic'
}

export function SetupScreen({ onStartGame, onOpenLab, initialSeatPolicies }: SetupScreenProps) {
  const initial = initialSeatPolicies ?? [DEFAULT_HUMAN, DEFAULT_HEURISTIC]
  const [playerCount, setPlayerCount] = useState(Math.max(2, Math.min(4, initial.length)))
  const [seats, setSeats] = useState<SeatPolicy[]>(() => {
    const out: SeatPolicy[] = []
    for (let i = 0; i < 4; i++) {
      out.push(initial[i] ?? (i === 0 ? DEFAULT_HUMAN : DEFAULT_HEURISTIC))
    }
    return out
  })
  const [labMembers, setLabMembers] = useState<LabMember[]>([])
  const [labReachable, setLabReachable] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchLeaderboard()
      .then(lb => {
        if (cancelled) return
        setLabMembers(lb.members)
        setLabReachable(true)
      })
      .catch(() => { if (!cancelled) setLabReachable(false) })
    return () => { cancelled = true }
  }, [])

  const toggleHuman = (i: number) => {
    setSeats(prev => {
      const next = prev.slice()
      next[i] = next[i].kind === 'human' ? DEFAULT_HEURISTIC : DEFAULT_HUMAN
      return next
    })
  }

  const setSeatKind = (i: number, kind: AIKindKey) => {
    setSeats(prev => {
      const next = prev.slice()
      next[i] = policyForKind(kind)
      return next
    })
  }

  const handleStart = () => {
    onStartGame(seats.slice(0, playerCount))
  }

  return (
    // Outer: vertical scroll when content overflows (landscape phone height
    // is ~414px and the full setup form is ~600px). Card is centered
    // horizontally via margin:auto; vertical centering is inline padding so
    // we never clip the top off-screen with `align-items:center` overflow.
    <div style={{
      height: '100%', overflowY: 'auto',
      padding: 'calc(12px + var(--sat)) calc(12px + var(--sal)) calc(12px + var(--sab)) calc(12px + var(--sar))',
      background: 'var(--paper)',
      WebkitOverflowScrolling: 'touch',
    }}>
      <div style={{
        background: 'var(--paper-dark)', padding: '18px 18px',
        width: '100%', maxWidth: 420, margin: '0 auto',
        border: '1px solid var(--rule)',
      }}>
        <h1 style={{
          margin: '0 0 2px', fontFamily: 'var(--font-display)',
          fontSize: 22, fontWeight: 700, color: 'var(--ink)',
          textAlign: 'center', letterSpacing: 0.5,
        }}>Tigris &amp; Euphrates</h1>
        <p style={{
          margin: '0 0 14px', fontFamily: 'var(--font-mono)',
          fontSize: 10, color: 'var(--ink-faint)', textAlign: 'center',
          textTransform: 'uppercase', letterSpacing: 2,
        }}>Mesopotamia · 3500 BC</p>

        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Players</label>
          <div style={{ display: 'flex', gap: 6 }}>
            {[2, 3, 4].map((n) => (
              <button
                key={n}
                onClick={() => setPlayerCount(n)}
                style={{
                  flex: 1, padding: '8px', minHeight: 40,
                  border: playerCount === n ? '1px solid var(--accent)' : '1px solid var(--rule)',
                  background: playerCount === n ? 'var(--paper-light)' : 'var(--paper)',
                  color: playerCount === n ? 'var(--accent)' : 'var(--ink-light)',
                  fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 700,
                  cursor: 'pointer',
                }}
              >{n}</button>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 12 }}>
          {Array.from({ length: playerCount }, (_, i) => {
            const dynasty = DYNASTIES[i]
            const seat = seats[i]
            const isHuman = seat.kind === 'human'
            return (
              <div key={i} style={{
                display: 'flex', flexDirection: 'column', gap: 4,
                padding: '6px 10px', background: 'var(--paper)',
                border: '1px solid var(--rule)', borderLeft: `3px solid ${dynasty.color}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontSize: 18 }}>{dynasty.symbol}</span>
                  <span style={{
                    flex: 1, fontFamily: 'var(--font-display)',
                    fontSize: 13, fontWeight: 700, color: dynasty.color,
                  }}>{dynasty.name}</span>
                  <button
                    onClick={() => toggleHuman(i)}
                    style={{
                      padding: '4px 10px', minHeight: 28, minWidth: 60,
                      border: '1px solid var(--rule)',
                      background: isHuman ? 'var(--paper-light)' : 'var(--paper-dark)',
                      color: isHuman ? dynasty.color : 'var(--ink-faint)',
                      fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 500,
                      textTransform: 'uppercase', letterSpacing: 1, cursor: 'pointer',
                    }}
                  >{isHuman ? 'Human' : 'AI'}</button>
                </div>
                {!isHuman && (
                  <select
                    value={kindForPolicy(seat)}
                    onChange={e => setSeatKind(i, e.target.value)}
                    style={{
                      width: '100%', padding: '4px 6px', minHeight: 28,
                      border: '1px solid var(--rule)', background: 'var(--paper)',
                      color: 'var(--ink-light)',
                      fontFamily: 'var(--font-mono)', fontSize: 10,
                    }}
                  >
                    <option value="heuristic">Heuristic</option>
                    <option value="onnx-default">Trained (browser, /policy.onnx)</option>
                    <option value="server">Trained (server)</option>
                    {labMembers.length > 0 && <optgroup label="Pool snapshots (lab server)">
                      {labMembers.map(m => (
                        <option key={m.name} value={m.name}>
                          {m.name}{m.elo != null ? ` · elo ${m.elo.toFixed(0)}` : ''}
                        </option>
                      ))}
                    </optgroup>}
                  </select>
                )}
              </div>
            )
          })}
        </div>

        <div style={{ display: 'flex', gap: 6 }}>
          <button
            onClick={handleStart}
            style={{
              flex: 2, padding: 10, minHeight: 40,
              border: '1px solid var(--accent)', background: 'var(--paper-light)',
              color: 'var(--accent)', fontFamily: 'var(--font-mono)',
              fontSize: 12, fontWeight: 500, cursor: 'pointer',
              textTransform: 'uppercase', letterSpacing: 2,
            }}
          >Start</button>

          <button
            onClick={onOpenLab}
            style={{
              flex: 1, padding: 10, minHeight: 40,
              border: '1px solid var(--rule)', background: 'var(--paper)',
              color: 'var(--ink-light)', fontFamily: 'var(--font-mono)',
              fontSize: 11, cursor: 'pointer',
              textTransform: 'uppercase', letterSpacing: 1,
            }}
          >Lab →</button>
        </div>

        <p style={{
          margin: '8px 0 0', fontFamily: 'var(--font-mono)', fontSize: 9,
          color: 'var(--ink-faint)', textAlign: 'center', letterSpacing: 0.5,
        }}>
          Lab server: {labReachable === null ? 'checking…' : labReachable ? `online (${labMembers.length} pool)` : 'offline (start python/policy_server.py)'}
        </p>
      </div>
    </div>
  )
}

const labelStyle: React.CSSProperties = {
  display: 'block', fontFamily: 'var(--font-mono)', fontSize: 10,
  color: 'var(--ink-faint)', textTransform: 'uppercase',
  letterSpacing: 1.5, marginBottom: 8,
}
