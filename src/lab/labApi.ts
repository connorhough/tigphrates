/**
 * Tiny client for the python/policy_server.py "lab" endpoints.
 * Browser → Python server → reads/writes models/pool/.
 */

// Derive the lab URL from the page's own host so the same build works for
// localhost AND remote viewers (Tailscale, LAN). The dev server runs on
// :5174, the lab on :8765 — both bound to 0.0.0.0.
function deriveLabBaseUrl(): string {
  if (typeof window === 'undefined') return 'http://localhost:8765'
  const { protocol, hostname } = window.location
  return `${protocol}//${hostname}:8765`
}

export const LAB_BASE_URL = deriveLabBaseUrl()

export interface LabMember {
  name: string
  path: string
  size_bytes: number
  mtime: number
  elo: number | null
  has_onnx: boolean
}

export interface Leaderboard {
  pool_dir: string
  agent_elo: number | null
  members: LabMember[]
}

export async function fetchLeaderboard(base: string = LAB_BASE_URL): Promise<Leaderboard> {
  const r = await fetch(`${base}/leaderboard`)
  if (!r.ok) throw new Error(`leaderboard ${r.status}`)
  return r.json()
}

export interface TrainJob {
  status: 'running' | 'done' | 'failed'
  exit_code: number | null
  started_at: number
  finished_at?: number
  time_budget: number
  log: string[]
}

export async function startTraining(timeBudget: number, env?: Record<string, string>, base: string = LAB_BASE_URL): Promise<{ job_id: string }> {
  const r = await fetch(`${base}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ time_budget: timeBudget, env }),
  })
  if (!r.ok) {
    const body = await r.json().catch(() => ({ error: r.statusText }))
    throw new Error(body.error || `train ${r.status}`)
  }
  return r.json()
}

export async function fetchJob(jobId: string, base: string = LAB_BASE_URL): Promise<TrainJob> {
  const r = await fetch(`${base}/train/${jobId}`)
  if (!r.ok) throw new Error(`job ${r.status}`)
  return r.json()
}
