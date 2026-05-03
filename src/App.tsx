import { useState, useMemo, useCallback } from 'react'
import { useGame } from './hooks/useGame'
import { GameBoard } from './components/GameBoard'
import { TopBar } from './components/TopBar'
import { PlayerPanel } from './components/PlayerPanel'
import { HandPanel } from './components/HandPanel'
import { ActionBar } from './components/ActionBar'
import { ConflictDialog } from './components/ConflictDialog'
import { MonumentDialog } from './components/MonumentDialog'
import { WarOrderDialog } from './components/WarOrderDialog'
import { GameOverScreen } from './components/GameOverScreen'
import { SetupScreen } from './components/SetupScreen'
import { LabScreen } from './components/LabScreen'
import { SwapDialog } from './components/SwapDialog'
import { getValidTilePlacements, getValidLeaderPlacements, canPlaceCatastrophe } from './engine/validation'
import { Position, LeaderColor, BOARD_ROWS, BOARD_COLS, GameState, GameAction } from './engine/types'
import { activePlayerIndex } from './bridge/encoder'
import { getAIAction } from './ai/simpleAI'
import { getRemoteAIAction } from './ai/remotePolicy'
import { SeatPolicy, DEFAULT_HEURISTIC, DEFAULT_HUMAN } from './types/seatPolicy'

type View = 'setup' | 'lab' | 'game'

function App() {
  const [view, setView] = useState<View>('setup')
  const [swapOpen, setSwapOpen] = useState(false)
  const [seatPolicies, setSeatPolicies] = useState<SeatPolicy[]>([
    DEFAULT_HUMAN, DEFAULT_HEURISTIC,
  ])

  // Per-seat policy dispatch. Uses the active player so commit-support
  // sub-phases route to the correct seat's AI (not state.currentPlayer).
  const policy = useCallback(
    async (s: GameState): Promise<GameAction> => {
      const idx = activePlayerIndex(s)
      const sp = seatPolicies[idx] ?? DEFAULT_HEURISTIC
      try {
        if (sp.kind === 'server') return await getRemoteAIAction(s)
        if (sp.kind === 'onnx') {
          const { getOnnxAIAction } = await import('./ai/onnxPolicy')
          return await getOnnxAIAction(s, sp.modelUrl)
        }
      } catch (err) {
        console.warn(`Seat ${idx} policy (${sp.kind}) failed, falling back to heuristic:`, err)
      }
      return getAIAction(s)
    },
    [seatPolicies],
  )

  const {
    state,
    dispatch,
    selectedTile,
    selectedLeader,
    placingCatastrophe,
    setSelectedTile,
    setSelectedLeader,
    setPlacingCatastrophe,
    startNewGame,
  } = useGame({ getAIAction: policy })

  const currentPlayer = state.players[state.currentPlayer]
  const isAITurn = !!currentPlayer.isAI
  const isActionPhase = state.turnPhase === 'action'

  const highlights = useMemo<Position[]>(() => {
    if (!isActionPhase) return []
    if (placingCatastrophe) {
      const valid: Position[] = []
      for (let row = 0; row < BOARD_ROWS; row++) {
        for (let col = 0; col < BOARD_COLS; col++) {
          if (canPlaceCatastrophe(state, { row, col })) {
            valid.push({ row, col })
          }
        }
      }
      return valid
    }
    if (selectedTile) {
      return getValidTilePlacements(state, selectedTile)
    }
    if (selectedLeader) {
      const currentLeader = currentPlayer.leaders.find(l => l.color === selectedLeader)
      const from = currentLeader?.position ?? null
      return getValidLeaderPlacements(state, selectedLeader, from)
    }
    return []
  }, [state, selectedTile, selectedLeader, placingCatastrophe, isActionPhase, currentPlayer.leaders])

  const handleCellClick = useCallback((pos: Position) => {
    if (!isActionPhase) return
    if (placingCatastrophe) {
      dispatch({ type: 'placeCatastrophe', position: pos, playerIndex: state.currentPlayer })
      setPlacingCatastrophe(false)
      return
    }
    if (selectedTile) {
      dispatch({ type: 'placeTile', color: selectedTile, position: pos, playerIndex: state.currentPlayer })
      setSelectedTile(null)
      return
    }
    if (selectedLeader) {
      dispatch({ type: 'placeLeader', color: selectedLeader, position: pos, playerIndex: state.currentPlayer })
      setSelectedLeader(null)
      return
    }
  }, [isActionPhase, placingCatastrophe, selectedTile, selectedLeader, state.currentPlayer, dispatch, setSelectedTile, setSelectedLeader, setPlacingCatastrophe])

  const handleOpenSwap = useCallback(() => setSwapOpen(true), [])
  const handleConfirmSwap = useCallback((indices: number[]) => {
    dispatch({ type: 'swapTiles', indices, playerIndex: state.currentPlayer })
    setSwapOpen(false)
  }, [state.currentPlayer, dispatch])
  const handleCancelSwap = useCallback(() => setSwapOpen(false), [])
  const handlePass = useCallback(() => {
    dispatch({ type: 'pass', playerIndex: state.currentPlayer })
  }, [state.currentPlayer, dispatch])
  const handleCommitSupport = useCallback((indices: number[]) => {
    const conflict = state.pendingConflict
    if (!conflict) return
    const committingPlayer = conflict.attackerCommitted === null
      ? conflict.attacker.playerIndex
      : conflict.defender.playerIndex
    dispatch({ type: 'commitSupport', indices, playerIndex: committingPlayer })
  }, [state.pendingConflict, dispatch])
  const handleBuildMonument = useCallback((monumentId: string) => {
    dispatch({ type: 'buildMonument', monumentId, playerIndex: state.currentPlayer })
  }, [state.currentPlayer, dispatch])
  const handleDeclineMonument = useCallback(() => {
    dispatch({ type: 'declineMonument', playerIndex: state.currentPlayer })
  }, [state.currentPlayer, dispatch])
  const handleChooseWarOrder = useCallback((color: LeaderColor) => {
    dispatch({ type: 'chooseWarOrder', color, playerIndex: state.currentPlayer })
  }, [state.currentPlayer, dispatch])
  const handleWithdrawLeader = useCallback((color: LeaderColor) => {
    dispatch({ type: 'withdrawLeader', color, playerIndex: state.currentPlayer })
  }, [state.currentPlayer, dispatch])

  const startGameWithPolicies = useCallback((sps: SeatPolicy[]) => {
    setSeatPolicies(sps)
    const aiFlags = sps.map(p => p.kind !== 'human')
    startNewGame(sps.length, aiFlags)
    setView('game')
  }, [startNewGame])

  if (view === 'setup') {
    return <SetupScreen
      onStartGame={startGameWithPolicies}
      onOpenLab={() => setView('lab')}
      initialSeatPolicies={seatPolicies}
    />
  }

  if (view === 'lab') {
    return <LabScreen
      onStartMatch={startGameWithPolicies}
      onBack={() => setView('setup')}
    />
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--paper)',
      overflow: 'hidden',
    }}>
      <TopBar state={state} />

      <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <GameBoard
          state={state}
          highlights={highlights}
          onCellClick={handleCellClick}
        />
        <PlayerPanel state={state} />
      </div>

      <HandPanel
        hand={currentPlayer.hand}
        selectedTile={selectedTile}
        onSelectTile={setSelectedTile}
        disabled={!isActionPhase || isAITurn}
      />

      <ActionBar
        state={state}
        selectedLeader={selectedLeader}
        placingCatastrophe={placingCatastrophe}
        onSelectLeader={setSelectedLeader}
        onPlaceCatastrophe={setPlacingCatastrophe}
        onSwapTiles={handleOpenSwap}
        onPass={handlePass}
        onWithdrawLeader={handleWithdrawLeader}
        disabled={!isActionPhase || isAITurn}
      />

      {state.turnPhase === 'conflictSupport' && state.pendingConflict && (
        <ConflictDialog
          state={state}
          currentViewingPlayer={state.currentPlayer}
          onCommitSupport={handleCommitSupport}
        />
      )}

      {state.turnPhase === 'monumentChoice' && state.pendingMonument && (
        <MonumentDialog
          state={state}
          onBuildMonument={handleBuildMonument}
          onDeclineMonument={handleDeclineMonument}
        />
      )}

      {state.turnPhase === 'warOrderChoice' && state.pendingConflict?.pendingWarColors && (
        <WarOrderDialog
          state={state}
          onChooseWarOrder={handleChooseWarOrder}
        />
      )}

      {state.turnPhase === 'gameOver' && (
        <GameOverScreen
          state={state}
          onPlayAgain={() => setView('setup')}
        />
      )}

      {swapOpen && !isAITurn && isActionPhase && (
        <SwapDialog
          hand={currentPlayer.hand}
          onConfirm={handleConfirmSwap}
          onCancel={handleCancelSwap}
        />
      )}
    </div>
  )
}

export default App
