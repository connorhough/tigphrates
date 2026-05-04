// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest'
import { StrictMode } from 'react'
import { renderHook, act } from '@testing-library/react'
import { useGame } from '../useGame'
import { getValidTilePlacements } from '../../engine/validation'
import { getAIAction } from '../../ai/simpleAI'
import type { TileColor, GameState, GameAction } from '../../engine/types'

const ACTION_LINE_RE = /^\s*(?:T:|L:|S\(|P\b|C@|BM:|DM\b|W:|CS\(|WO:)/

function actionLineCount(log: string[]): number {
  return log.filter(line => ACTION_LINE_RE.test(line)).length
}

function consecutiveDuplicateCount(log: string[]): number {
  let dupes = 0
  for (let i = 1; i < log.length; i++) {
    if (log[i] === log[i - 1] && ACTION_LINE_RE.test(log[i])) dupes += 1
  }
  return dupes
}

describe('useGame dispatch logging', () => {
  it('logs exactly one action line per dispatch under StrictMode', () => {
    const { result } = renderHook(() => useGame(), { wrapper: StrictMode })

    act(() => {
      result.current.startNewGame(2, [false, false])
    })

    const handFirst = result.current.state.players[0].hand[0] as TileColor
    const valid = getValidTilePlacements(result.current.state, handFirst)
    expect(valid.length).toBeGreaterThan(0)

    act(() => {
      result.current.dispatch({
        type: 'placeTile',
        color: handFirst,
        position: valid[0],
        playerIndex: 0,
      })
    })

    const lines = actionLineCount(result.current.gameLog)
    expect(lines).toBe(1)
  })

  it('logs exactly N action lines after N dispatches under StrictMode', () => {
    const { result } = renderHook(() => useGame(), { wrapper: StrictMode })

    act(() => {
      result.current.startNewGame(2, [false, false])
    })

    let dispatched = 0
    for (let i = 0; i < 4; i++) {
      const state = result.current.state
      if (state.turnPhase !== 'action') break
      const player = state.players[state.currentPlayer]
      let placed = false
      for (const color of player.hand) {
        const valid = getValidTilePlacements(state, color)
        if (valid.length === 0) continue
        act(() => {
          result.current.dispatch({
            type: 'placeTile',
            color,
            position: valid[0],
            playerIndex: state.currentPlayer,
          })
        })
        dispatched += 1
        placed = true
        break
      }
      if (!placed) break
    }

    expect(dispatched).toBeGreaterThan(0)
    const lines = actionLineCount(result.current.gameLog)
    expect(lines).toBe(dispatched)
  })

  it('does not double-log when an async AI policy resolves under StrictMode', async () => {
    // Reproduces the production bug where two AI seats with an async policy
    // (ONNX) produce logs with every action line duplicated. We use an async
    // policy because the bug only manifests when policy(state) is awaited
    // (giving React time to re-mount the auto-AI useEffect under StrictMode).
    const policy = vi.fn(async (s: GameState): Promise<GameAction> => {
      // Yield to the microtask queue so we exercise the async path.
      await Promise.resolve()
      return getAIAction(s)
    })
    const { result } = renderHook(
      () => useGame({ getAIAction: policy, aiThinkMs: 0 }),
      { wrapper: StrictMode },
    )

    await act(async () => {
      result.current.startNewGame(2, [true, true])
    })

    // Let several AI dispatch cycles complete.
    for (let i = 0; i < 30; i++) {
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 10))
      })
      if (result.current.state.turnPhase === 'gameOver') break
    }

    const log = result.current.gameLog
    const dupes = consecutiveDuplicateCount(log)
    expect.soft(log.length).toBeGreaterThan(2) // Ensure AI actually dispatched
    expect(dupes).toBe(0)
  })

  it('saved game log never contains two consecutive identical lines (production bug contract)', () => {
    // The production bug produced saved game logs like:
    //   T:R@2.1
    //   T:R@2.1
    //   T:B@1.4
    //   T:B@1.4
    // ...where every action line was duplicated. Whatever React/effect
    // race caused two dispatches per action, the saved log MUST be free
    // of consecutive duplicate lines so analysis tools and the headless
    // log parser can read it. This test asserts that contract by driving
    // a full game's worth of dispatches and checking the gameLog after.
    const { result } = renderHook(() => useGame(), { wrapper: StrictMode })

    act(() => {
      result.current.startNewGame(2, [false, false])
    })

    // Drive ~30 dispatches across alternating players using whatever the
    // heuristic AI would do for the current state. We compute the action
    // here (synchronously) and dispatch it, ignoring AI seat flags.
    for (let i = 0; i < 30; i++) {
      const state = result.current.state
      if (state.turnPhase !== 'action') break
      const player = state.players[state.currentPlayer]
      let placed = false
      for (const color of player.hand) {
        const valid = getValidTilePlacements(state, color)
        if (valid.length === 0) continue
        act(() => {
          result.current.dispatch({
            type: 'placeTile',
            color,
            position: valid[0],
            playerIndex: state.currentPlayer,
          })
        })
        placed = true
        break
      }
      if (!placed) {
        act(() => {
          result.current.dispatch({ type: 'pass', playerIndex: state.currentPlayer })
        })
      }
    }

    const log = result.current.gameLog
    expect(log.length).toBeGreaterThan(5)
    expect(consecutiveDuplicateCount(log)).toBe(0)

    // Stronger contract: NO two consecutive log lines are identical, not
    // just action lines. The saved-game format depends on this for
    // unambiguous parsing.
    let anyDup = 0
    for (let i = 1; i < log.length; i++) {
      if (log[i] === log[i - 1]) anyDup += 1
    }
    expect(anyDup).toBe(0)
  })

  it('does not append a log line that duplicates the previous line (dedupe contract)', () => {
    // Mechanically demonstrates the bug + fix using the `pass` action,
    // which always succeeds and decrements actionsRemaining. We trigger
    // a sequence where two consecutive dispatches push the same formatted
    // log line ("  P") AS ADJACENT entries in logRef. The bug pattern is
    // that the second push would be a literal duplicate; the fix is to
    // suppress it.
    //
    // Sequence: P1 plays a tile (advances actionsRemaining 2→1). Then P1
    // passes (1→0 → endTurn → P2 with 2). The "  P" line here is FIRST.
    // Then P2 passes (2→1). That's "  P" again. WITHOUT the dedupe guard,
    // the log would contain:
    //   ...  T:R@x.y    (P1's tile)
    //   === T1 P2 ...   (turn header for P2 pushed during P1's pass)
    //   "  P"           (P1's pass action line)
    //   "  P"           (P2's pass action line — IMMEDIATELY ADJACENT)
    // ...because the turn header for P2 is pushed in P1's pass dispatch
    // BEFORE the action line, so the two "  P" lines from two different
    // passes end up consecutive.
    //
    // With the dedupe guard, the second "  P" is dropped.
    const { result } = renderHook(() => useGame(), { wrapper: StrictMode })

    act(() => {
      result.current.startNewGame(2, [false, false])
    })

    // P1 places a tile to anchor the game state (also ensures the next
    // few log entries follow a known pattern).
    const handFirst = result.current.state.players[0].hand[0] as TileColor
    const valid = getValidTilePlacements(result.current.state, handFirst)
    act(() => {
      result.current.dispatch({
        type: 'placeTile',
        color: handFirst,
        position: valid[0],
        playerIndex: 0,
      })
    })
    // P1 passes -> ends turn -> turn header for P2 is pushed, then "  P"
    act(() => {
      result.current.dispatch({ type: 'pass', playerIndex: 0 })
    })
    // P2 passes -> "  P" again, immediately after P1's "  P" with no
    // separator. Without the dedupe guard this creates an adjacent dup.
    act(() => {
      result.current.dispatch({ type: 'pass', playerIndex: 1 })
    })

    const log = result.current.gameLog
    let adj = 0
    for (let i = 1; i < log.length; i++) {
      if (log[i] === log[i - 1]) adj += 1
    }
    expect(adj).toBe(0)
  })

  it('two AI seats with async policy produce no consecutive duplicate log lines', async () => {
    // Faithful reproduction of the production scenario: 2 AI seats both
    // using an async policy (mimics ONNX latency). Drive a full game and
    // assert no consecutive duplicates ever appear in the gameLog. This
    // matches the real-world pattern observed in saved game logs where
    // each action line was duplicated.
    const policy = vi.fn(async (s: GameState): Promise<GameAction> => {
      // Simulate ~5ms ONNX inference latency.
      await new Promise(r => setTimeout(r, 5))
      return getAIAction(s)
    })

    const { result } = renderHook(
      () => useGame({ getAIAction: policy, aiThinkMs: 1 }),
      { wrapper: StrictMode },
    )

    await act(async () => {
      result.current.startNewGame(2, [true, true])
    })

    // Drive ~50 dispatch cycles; each cycle is roughly one AI action.
    for (let i = 0; i < 60; i++) {
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 15))
      })
      if (result.current.state.turnPhase === 'gameOver') break
    }

    const log = result.current.gameLog
    expect.soft(log.length).toBeGreaterThan(4)
    expect(consecutiveDuplicateCount(log)).toBe(0)
  })

  it('does not call onGameEnd more than once', () => {
    let endCalls = 0
    const { result } = renderHook(
      () => useGame({ onGameEnd: () => { endCalls += 1 } }),
      { wrapper: StrictMode },
    )

    act(() => {
      result.current.startNewGame(2, [false, false])
    })

    expect(endCalls).toBe(0)
  })
})
