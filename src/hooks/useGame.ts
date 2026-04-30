import { useState, useCallback, useEffect } from 'react'
import { gameReducer } from '../engine/reducer'
import { createGame } from '../engine/setup'
import { GameState, GameAction, TileColor, LeaderColor } from '../engine/types'
import { getAIAction } from '../ai/simpleAI'

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
}

export type AIPolicy = (state: GameState) => GameAction | Promise<GameAction>

interface UseGameOptions {
  // Override the AI policy. Defaults to the heuristic in src/ai/simpleAI.
  // Use to plug in a trained model via a bridge call without touching this file.
  getAIAction?: AIPolicy
  aiThinkMs?: number
}

export function useGame(options: UseGameOptions = {}): UseGameReturn {
  const policy: AIPolicy = options.getAIAction ?? getAIAction
  const aiThinkMs = options.aiThinkMs ?? 500
  const [state, setState] = useState<GameState>(() => createGame(2))
  const [selectedTile, setSelectedTile] = useState<TileColor | null>(null)
  const [selectedLeader, setSelectedLeader] = useState<LeaderColor | null>(null)
  const [placingCatastrophe, setPlacingCatastrophe] = useState(false)

  const dispatch = useCallback((action: GameAction & { playerIndex: number }) => {
    setState(prev => {
      try {
        return gameReducer(prev, action)
      } catch (err) {
        console.error('Game action failed:', err)
        return prev
      }
    })
  }, [])

  const startNewGame = useCallback((playerCount: number, aiFlags?: boolean[]) => {
    setState(createGame(playerCount, aiFlags))
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

  // Auto-trigger AI turns
  useEffect(() => {
    const player = state.players[state.currentPlayer]
    if (!player.isAI) return
    if (state.turnPhase === 'gameOver') return

    let cancelled = false
    const timeout = setTimeout(async () => {
      try {
        const action = await policy(state)
        if (cancelled) return
        dispatch({ ...action, playerIndex: state.currentPlayer })
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
  }
}
