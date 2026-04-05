import { describe, it, expect } from 'vitest'
import { runGame, runTournament } from '../runGame'

describe('runGame', () => {
  it('completes a 2-player game', () => {
    const result = runGame(2)

    expect(result.winner).toBeGreaterThanOrEqual(0)
    expect(result.winner).toBeLessThan(2)
    expect(result.scores).toHaveLength(2)
    expect(result.turnCount).toBeGreaterThan(0)
    expect(['gameOver', 'maxTurns']).toContain(result.reason)
  })

  it('completes a 3-player game', () => {
    const result = runGame(3)

    expect(result.scores).toHaveLength(3)
    expect(result.winner).toBeGreaterThanOrEqual(0)
    expect(result.winner).toBeLessThan(3)
  })

  it('completes a 4-player game', () => {
    const result = runGame(4)

    expect(result.scores).toHaveLength(4)
    expect(result.winner).toBeGreaterThanOrEqual(0)
    expect(result.winner).toBeLessThan(4)
  })

  it('returns valid score structures', () => {
    const result = runGame(2)

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

  it('respects maxActions parameter', () => {
    const result = runGame(2, 100)

    expect(result.turnCount).toBeLessThanOrEqual(100)
  })
})

describe('runTournament', () => {
  it('runs multiple games and tracks wins', () => {
    const { results, wins, avgMinScores } = runTournament(2, 2, 500)

    expect(results).toHaveLength(2)
    expect(wins).toHaveLength(2)
    expect(avgMinScores).toHaveLength(2)

    // Total wins should equal game count
    expect(wins[0] + wins[1]).toBe(2)
  }, 30000)

  it('calculates average min scores', () => {
    const { avgMinScores } = runTournament(2, 2, 500)

    for (const avg of avgMinScores) {
      expect(avg).toBeGreaterThanOrEqual(0)
      expect(Number.isFinite(avg)).toBe(true)
    }
  }, 30000)
})
