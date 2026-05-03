import { useEffect, useRef, useState } from 'react'
import { SeatPolicy } from '../types/seatPolicy'
import {
  fetchLeaderboard, fetchJob, startTraining, Leaderboard, LabMember, TrainJob,
  LAB_BASE_URL,
} from '../lab/labApi'

interface LabScreenProps {
  onStartMatch: (seatPolicies: SeatPolicy[]) => void
  onBack: () => void
}

export function LabScreen({ onStartMatch, onBack }: LabScreenProps) {
  const [lb, setLb] = useState<Leaderboard | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [timeBudget, setTimeBudget] = useState<number>(300)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<TrainJob | null>(null)
  const [matchA, setMatchA] = useState<string>('')
  const [matchB, setMatchB] = useState<string>('')
  const logEndRef = useRef<HTMLDivElement | null>(null)

  const refresh = async () => {
    try {
      const l = await fetchLeaderboard()
      setLb(l)
      setError(null)
      // Auto-pick top two for the match form if empty.
      if (!matchA && l.members[0]) setMatchA(l.members[0].name)
      if (!matchB && l.members[1]) setMatchB(l.members[1].name)
    } catch (e) {
      setError(`${e}`)
    }
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { refresh() }, [])

  // Poll the active training job + refresh leaderboard.
  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    const tick = async () => {
      try {
        const j = await fetchJob(jobId)
        if (cancelled) return
        setJob(j)
        if (j.status !== 'running') {
          // Training finished — pull fresh leaderboard once.
          refresh()
          return
        }
        setTimeout(tick, 2000)
        // Side-effect refresh during training so new pool members appear.
        refresh()
      } catch {
        if (!cancelled) setTimeout(tick, 4000)
      }
    }
    tick()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  // Auto-scroll log to bottom as new lines arrive.
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'auto' })
  }, [job?.log.length])

  const handleStartTraining = async () => {
    try {
      const r = await startTraining(timeBudget)
      setJobId(r.job_id)
      setJob(null)
    } catch (e) {
      setError(`${e}`)
    }
  }

  const handleWatchMatch = () => {
    if (!matchA || !matchB) return
    const url = (n: string) => `${LAB_BASE_URL}/pool/${n}.onnx`
    onStartMatch([
      { kind: 'onnx', modelUrl: url(matchA) },
      { kind: 'onnx', modelUrl: url(matchB) },
    ])
  }

  const isRunning = job?.status === 'running'

  return (
    <div style={{
      height: '100%', overflowY: 'auto', background: 'var(--paper)',
      padding: 'calc(20px + var(--sat)) calc(20px + var(--sal)) calc(20px + var(--sab)) calc(20px + var(--sar))',
    }}>
      <div style={{ maxWidth: 720, margin: '0 auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
          <button onClick={onBack} style={btnGhost}>← Back</button>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 22, color: 'var(--ink)' }}>Lab</h1>
          <div style={{ flex: 1 }} />
          <button onClick={refresh} style={btnGhost}>Refresh</button>
        </div>

        {error && <div style={{
          padding: 10, marginBottom: 16,
          border: '1px solid var(--rule)', background: 'var(--paper-dark)',
          fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-faint)',
        }}>
          Server error: {error}. Start it with <code>python python/policy_server.py</code>.
        </div>}

        {/* --- Leaderboard --- */}
        <section style={section}>
          <h2 style={h2}>Leaderboard</h2>
          {lb && lb.agent_elo != null && (
            <p style={{
              margin: '0 0 10px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-faint)',
            }}>Agent (latest): elo {lb.agent_elo.toFixed(1)}</p>
          )}
          {lb && lb.members.length === 0 && (
            <p style={muted}>Pool is empty. Start a training run to populate it.</p>
          )}
          {lb && lb.members.length > 0 && (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--rule)' }}>
                  <th style={th}>#</th>
                  <th style={th}>Snapshot</th>
                  <th style={thRight}>Persistent Elo</th>
                  <th style={thRight}>Size (KB)</th>
                  <th style={thRight}>Saved</th>
                  <th style={th}>ONNX</th>
                </tr>
              </thead>
              <tbody>
                {lb.members.map((m: LabMember, i: number) => (
                  <tr key={m.name} style={{ borderBottom: '1px solid var(--rule)' }}>
                    <td style={td}>{i + 1}</td>
                    <td style={td}>{m.name}</td>
                    <td style={tdRight}>{m.elo != null ? m.elo.toFixed(1) : '—'}</td>
                    <td style={tdRight}>{(m.size_bytes / 1024).toFixed(0)}</td>
                    <td style={tdRight}>{relTime(m.mtime)}</td>
                    <td style={td}>{m.has_onnx ? '✓' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* --- Train --- */}
        <section style={section}>
          <h2 style={h2}>Train</h2>
          <p style={muted}>Spawns python/train.py with TIME_BUDGET seconds. Mac mini constraint: one job at a time. New pool snapshots appear in the leaderboard above.</p>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
            <label style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-faint)' }}>Budget (sec)</label>
            <input
              type="number"
              min={10} max={7200} step={30}
              value={timeBudget}
              onChange={e => setTimeBudget(parseInt(e.target.value || '300', 10))}
              disabled={isRunning}
              style={inputStyle}
            />
            <button
              onClick={handleStartTraining}
              disabled={isRunning}
              style={isRunning ? btnDisabled : btnAccent}
            >{isRunning ? 'Running…' : 'Start training'}</button>
            {jobId && <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-faint)' }}>
              job {jobId} · {job?.status ?? 'starting'}{job?.exit_code != null ? ` · exit ${job.exit_code}` : ''}
            </span>}
          </div>
          {job && job.log.length > 0 && (
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 10, lineHeight: 1.4,
              background: 'var(--paper-dark)', border: '1px solid var(--rule)',
              padding: 10, height: 180, overflowY: 'auto', color: 'var(--ink-light)',
              whiteSpace: 'pre-wrap',
            }}>
              {job.log.map((line, i) => <div key={i}>{line}</div>)}
              <div ref={logEndRef} />
            </div>
          )}
        </section>

        {/* --- Watch match --- */}
        <section style={section}>
          <h2 style={h2}>Watch match</h2>
          <p style={muted}>Pit two snapshots against each other in a 2-player game. Each side fetches its ONNX file from /pool/&lt;name&gt;.onnx (lazily exported on first request).</p>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <label style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-faint)' }}>Seat 1</label>
            <select value={matchA} onChange={e => setMatchA(e.target.value)} style={inputStyle}>
              <option value="">—</option>
              {lb?.members.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
            <label style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--ink-faint)' }}>vs Seat 2</label>
            <select value={matchB} onChange={e => setMatchB(e.target.value)} style={inputStyle}>
              <option value="">—</option>
              {lb?.members.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
            <button
              onClick={handleWatchMatch}
              disabled={!matchA || !matchB}
              style={!matchA || !matchB ? btnDisabled : btnAccent}
            >Watch →</button>
          </div>
        </section>
      </div>
    </div>
  )
}

function relTime(mtimeSec: number): string {
  const dtSec = Math.max(0, Date.now() / 1000 - mtimeSec)
  if (dtSec < 60) return `${Math.floor(dtSec)}s ago`
  if (dtSec < 3600) return `${Math.floor(dtSec / 60)}m ago`
  if (dtSec < 86400) return `${Math.floor(dtSec / 3600)}h ago`
  return `${Math.floor(dtSec / 86400)}d ago`
}

const section: React.CSSProperties = {
  background: 'var(--paper-dark)', border: '1px solid var(--rule)',
  padding: '16px 18px', marginBottom: 16,
}
const h2: React.CSSProperties = {
  margin: '0 0 12px', fontFamily: 'var(--font-display)', fontSize: 14,
  color: 'var(--ink)', textTransform: 'uppercase', letterSpacing: 1.5,
}
const muted: React.CSSProperties = {
  margin: '0 0 12px', fontFamily: 'var(--font-mono)', fontSize: 10,
  color: 'var(--ink-faint)', lineHeight: 1.4,
}
const th: React.CSSProperties = { textAlign: 'left', padding: '6px 8px', color: 'var(--ink-faint)', fontWeight: 500 }
const thRight: React.CSSProperties = { ...th, textAlign: 'right' }
const td: React.CSSProperties = { padding: '6px 8px', color: 'var(--ink-light)' }
const tdRight: React.CSSProperties = { ...td, textAlign: 'right' }
const btnGhost: React.CSSProperties = {
  padding: '6px 12px', minHeight: 32,
  border: '1px solid var(--rule)', background: 'var(--paper)',
  color: 'var(--ink-light)', fontFamily: 'var(--font-mono)', fontSize: 11,
  cursor: 'pointer',
}
const btnAccent: React.CSSProperties = {
  padding: '8px 16px', minHeight: 36,
  border: '1px solid var(--accent)', background: 'var(--paper-light)',
  color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: 11,
  cursor: 'pointer', textTransform: 'uppercase', letterSpacing: 1,
}
const btnDisabled: React.CSSProperties = {
  ...btnAccent, opacity: 0.5, cursor: 'not-allowed',
}
const inputStyle: React.CSSProperties = {
  padding: '6px 10px', minHeight: 32,
  border: '1px solid var(--rule)', background: 'var(--paper)',
  color: 'var(--ink-light)', fontFamily: 'var(--font-mono)', fontSize: 11,
  minWidth: 120,
}
