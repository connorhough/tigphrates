import { describe, it, expect } from 'vitest'
import { runGame, runTournament } from '../runGame'

describe('runGame', () => {
  it('completes a 2-player game', async () => {
    const result = await runGame({ playerCount: 2, maxActions: 500 })

    expect(result.winner).toBeGreaterThanOrEqual(0)
    expect(result.winner).toBeLessThan(2)
    expect(result.scores).toHaveLength(2)
    expect(result.turnCount).toBeGreaterThan(0)
    expect(result.turns).toBeGreaterThan(0)
    expect(['gameOver', 'maxTurns']).toContain(result.reason)
  })

  it('completes a 3-player game', async () => {
    const result = await runGame({ playerCount: 3, maxActions: 500 })

    expect(result.scores).toHaveLength(3)
    expect(result.winner).toBeGreaterThanOrEqual(0)
    expect(result.winner).toBeLessThan(3)
  })

  it('completes a 4-player game', async () => {
    const result = await runGame({ playerCount: 4, maxActions: 500 })

    expect(result.scores).toHaveLength(4)
    expect(result.winner).toBeGreaterThanOrEqual(0)
    expect(result.winner).toBeLessThan(4)
  })

  it('returns valid score structures', async () => {
    const result = await runGame({ playerCount: 2, maxTurns: 5 })

    for (const score of result.scores) {
      expect(score.dynasty).toBeTruthy()
      expect(score.minScore).toBeGreaterThanOrEqual(0)
      expect(score.score).toHaveProperty('red')
      expect(score.score).toHaveProperty('blue')
      expect(score.score).toHaveProperty('green')
      expect(score.score).toHaveProperty('black')
      expect(score.treasures).toBeGreaterThanOrEqual(0)
    }
  })

  it('respects maxTurns parameter', async () => {
    const result = await runGame({ playerCount: 2, maxTurns: 3 })

    expect(result.turns).toBeLessThanOrEqual(4) // may overshoot by 1 due to counting
    expect(result.reason).toBe('turnLimit')
  })

  it('respects legacy positional arguments', async () => {
    const result = await runGame(2, 100)

    expect(result.turnCount).toBeLessThanOrEqual(100)
  })
})

describe('game log', () => {
  it('produces log lines by default', async () => {
    const result = await runGame({ playerCount: 2, maxTurns: 2 })

    expect(result.log.length).toBeGreaterThan(0)
    expect(result.log[0]).toMatch(/^GAME 2p/)
  })

  it('includes turn headers', async () => {
    const result = await runGame({ playerCount: 2, maxTurns: 3 })

    const turnHeaders = result.log.filter(l => l.startsWith('=== T'))
    expect(turnHeaders.length).toBeGreaterThan(0)
    expect(turnHeaders[0]).toMatch(/=== T\d+ P\d+\(\w+\) hand=/)
  })

  it('includes final summary', async () => {
    const result = await runGame({ playerCount: 2, maxTurns: 2 })

    const endLine = result.log.find(l => l.startsWith('--- END:'))
    expect(endLine).toBeDefined()
    const finalLine = result.log.find(l => l.startsWith('FINAL:'))
    expect(finalLine).toBeDefined()
  })

  it('logs score deltas on scoring actions', async () => {
    const result = await runGame({ playerCount: 2, maxTurns: 5 })

    // At least some actions should produce score deltas
    const withDeltas = result.log.filter(l => l.includes('[P'))
    expect(withDeltas.length).toBeGreaterThan(0)
  })

  it('can be disabled', async () => {
    const result = await runGame({ playerCount: 2, maxTurns: 2, log: false })

    expect(result.log).toHaveLength(0)
  })

  it('logs score snapshots at interval', async () => {
    const result = await runGame({ playerCount: 2, maxTurns: 12 })

    const snapshots = result.log.filter(l => l.includes('SCORES:'))
    expect(snapshots.length).toBeGreaterThan(0)
  })
})

describe('runTournament', () => {
  it('runs multiple games and tracks wins', async () => {
    const { results, wins, avgMinScores } = await runTournament(2, 2, 500)

    expect(results).toHaveLength(2)
    expect(wins).toHaveLength(2)
    expect(avgMinScores).toHaveLength(2)

    expect(wins[0] + wins[1]).toBe(2)
  }, 30000)

  it('calculates average min scores', async () => {
    const { avgMinScores } = await runTournament(2, 2, 500)

    for (const avg of avgMinScores) {
      expect(avg).toBeGreaterThanOrEqual(0)
      expect(Number.isFinite(avg)).toBe(true)
    }
  }, 30000)

  it('tournament games have no logs by default', async () => {
    const { results } = await runTournament(2, 2, 500)

    for (const r of results) {
      expect(r.log).toHaveLength(0)
    }
  }, 30000)
})
