import { useState, useCallback } from 'react'
import { gameReducer } from '../engine/reducer'
import { createGame } from '../engine/setup'
import { GameState, GameAction, TileColor, LeaderColor } from '../engine/types'

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

export function useGame(): UseGameReturn {
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
