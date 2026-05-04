import { useState, useCallback, useEffect, useRef } from 'react'
import { gameReducer } from '../engine/reducer'
import { createGame } from '../engine/setup'
import { GameState, GameAction, TileColor, LeaderColor } from '../engine/types'
import { getAIAction } from '../ai/simpleAI'
import { activePlayerIndex } from '../bridge/encoder'
import {
  formatActionLine, conflictResolutionLine, handSummary, scoreSnapshot,
} from '../engine/logFormat'

/**
 * Appends `line` to `log` unless the last line is byte-identical. This
 * defends against an upstream race where the same dispatch — and thus the
 * same formatted action line — is recorded twice. The race itself was
 * elusive to reproduce in jsdom (production was browser + StrictMode +
 * async ONNX policy) so this guard exists at the log-write site to ensure
 * saved game logs are unambiguous, regardless of which React/effect quirk
 * caused the duplicate dispatch.
 */
function pushUnique(log: string[], line: string): void {
  if (log.length > 0 && log[log.length - 1] === line) return
  log.push(line)
}

interface UseGameReturn {
  state: GameState
  dispatch: (action: GameAction & { playerIndex: number }) => void
  selectedTile: TileColor | null
  selectedLeader: LeaderColor | null
  placingCatastrophe: boolean
  setSelectedTile: (color: TileColor | null) => void
  setSelectedLeader: (color: LeaderColor | null) => void
  setPlacingCatastrophe: (v: boolean) => void
  startNewGame: (playerCount: number, aiFlags?: boolean[]) => void
  /** Compact game log accumulated since the last startNewGame. */
  gameLog: string[]
}

export type AIPolicy = (state: GameState) => GameAction | Promise<GameAction>

interface UseGameOptions {
  // Override the AI policy. Defaults to the heuristic in src/ai/simpleAI.
  // Use to plug in a trained model via a bridge call without touching this file.
  getAIAction?: AIPolicy
  aiThinkMs?: number
  /** Called once per finished game with the full compact log. */
  onGameEnd?: (log: string[], finalState: GameState) => void
}

export function useGame(options: UseGameOptions = {}): UseGameReturn {
  const policy: AIPolicy = options.getAIAction ?? getAIAction
  const aiThinkMs = options.aiThinkMs ?? 500
  const onGameEnd = options.onGameEnd
  const [state, setState] = useState<GameState>(() => createGame(2))
  const [selectedTile, setSelectedTile] = useState<TileColor | null>(null)
  const [selectedLeader, setSelectedLeader] = useState<LeaderColor | null>(null)
  const [placingCatastrophe, setPlacingCatastrophe] = useState(false)
  // Mirror of `state` updated synchronously in dispatch so the dispatcher
  // can compute `next` outside setState's updater. Side effects (log
  // mutation, onGameEnd hook) live in dispatch's body — running once per
  // dispatch — instead of inside a setState updater that StrictMode
  // intentionally invokes twice.
  const stateRef = useRef<GameState>(state)
  const logRef = useRef<string[]>([])
  const lastTurnPlayerRef = useRef<number>(-1)
  const turnNumberRef = useRef<number>(0)
  const gameEndedRef = useRef<boolean>(false)
  const onGameEndRef = useRef(onGameEnd)
  useEffect(() => { onGameEndRef.current = onGameEnd }, [onGameEnd])
  const [gameLog, setGameLog] = useState<string[]>([])

  const dispatch = useCallback((action: GameAction & { playerIndex: number }) => {
    const prev = stateRef.current
    let next: GameState
    try {
      next = gameReducer(prev, action)
    } catch (err) {
      console.error('Game action failed:', err)
      return
    }

    if (
      next.turnPhase === 'action' &&
      next.actionsRemaining === 2 &&
      next.currentPlayer !== lastTurnPlayerRef.current
    ) {
      if (
        next.currentPlayer <= lastTurnPlayerRef.current ||
        lastTurnPlayerRef.current === -1
      ) {
        turnNumberRef.current += 1
      }
      lastTurnPlayerRef.current = next.currentPlayer
      const p = next.players[next.currentPlayer]
      logRef.current.push(
        `=== T${turnNumberRef.current} P${next.currentPlayer + 1}(${p.dynasty}) hand=${handSummary(p.hand)} bag=${next.bag.length} ===`,
      )
    }

    const { playerIndex: _ignore, ...gameAction } = action
    pushUnique(logRef.current, formatActionLine(gameAction as GameAction, prev, next))
    const conflict = conflictResolutionLine(prev, next)
    if (conflict) pushUnique(logRef.current, conflict)

    if (next.turnPhase === 'gameOver' && !gameEndedRef.current) {
      gameEndedRef.current = true
      logRef.current.push(`--- END after ${turnNumberRef.current} turns ---`)
      logRef.current.push(`FINAL: ${scoreSnapshot(next.players)}`)
      const snapshot = logRef.current.slice()
      setTimeout(() => onGameEndRef.current?.(snapshot, next), 0)
    }

    stateRef.current = next
    setState(next)
    setGameLog(logRef.current.slice())
  }, [])

  const startNewGame = useCallback((playerCount: number, aiFlags?: boolean[]) => {
    logRef.current = []
    lastTurnPlayerRef.current = -1
    turnNumberRef.current = 0
    gameEndedRef.current = false
    const initial = createGame(playerCount, aiFlags)
    logRef.current.push(`GAME ${playerCount}p | bag=${initial.bag.length}`)
    for (let i = 0; i < playerCount; i++) {
      const p = initial.players[i]
      logRef.current.push(
        `  P${i + 1}(${p.dynasty}) ${aiFlags?.[i] ? 'AI' : 'human'} hand=${handSummary(p.hand)}`,
      )
    }
    stateRef.current = initial
    setState(initial)
    setGameLog(logRef.current.slice())
    setSelectedTile(null)
    setSelectedLeader(null)
    setPlacingCatastrophe(false)
  }, [])

  const handleSetSelectedTile = useCallback((color: TileColor | null) => {
    setSelectedTile(color)
    setSelectedLeader(null)
    setPlacingCatastrophe(false)
  }, [])

  const handleSetSelectedLeader = useCallback((color: LeaderColor | null) => {
    setSelectedLeader(color)
    setSelectedTile(null)
    setPlacingCatastrophe(false)
  }, [])

  const handleSetPlacingCatastrophe = useCallback((v: boolean) => {
    setPlacingCatastrophe(v)
    if (v) {
      setSelectedTile(null)
      setSelectedLeader(null)
    }
  }, [])

  // Auto-trigger AI turns. The "active player" depends on the phase: in
  // conflictSupport it's whichever side hasn't committed (often the
  // defender, NOT state.currentPlayer). Dispatching with the wrong
  // playerIndex would make the reducer throw and the trigger would loop
  // forever, so we compute the active player here too.
  useEffect(() => {
    if (state.turnPhase === 'gameOver') return
    const activeIdx = activePlayerIndex(state)
    const player = state.players[activeIdx]
    if (!player.isAI) return

    let cancelled = false
    const timeout = setTimeout(async () => {
      try {
        const action = await policy(state)
        if (cancelled) return
        dispatch({ ...action, playerIndex: activeIdx })
      } catch (err) {
        console.error('AI policy failed:', err)
      }
    }, aiThinkMs)

    return () => {
      cancelled = true
      clearTimeout(timeout)
    }
  }, [state, dispatch, policy, aiThinkMs])

  return {
    state,
    dispatch,
    selectedTile,
    selectedLeader,
    placingCatastrophe,
    setSelectedTile: handleSetSelectedTile,
    setSelectedLeader: handleSetSelectedLeader,
    setPlacingCatastrophe: handleSetPlacingCatastrophe,
    startNewGame,
    gameLog,
  }
}
