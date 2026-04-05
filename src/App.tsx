import { useMemo, useCallback } from 'react'
import { useGame } from './hooks/useGame'
import { GameBoard } from './components/GameBoard'
import { TopBar } from './components/TopBar'
import { PlayerPanel } from './components/PlayerPanel'
import { HandPanel } from './components/HandPanel'
import { ActionBar } from './components/ActionBar'
import { ConflictDialog } from './components/ConflictDialog'
import { MonumentDialog } from './components/MonumentDialog'
import { WarOrderDialog } from './components/WarOrderDialog'
import { getValidTilePlacements, getValidLeaderPlacements, canPlaceCatastrophe } from './engine/validation'
import { Position, LeaderColor, BOARD_ROWS, BOARD_COLS } from './engine/types'

function App() {
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
  } = useGame()

  const currentPlayer = state.players[state.currentPlayer]
  const isActionPhase = state.turnPhase === 'action'

  // Compute valid placements for highlighting
  const highlights = useMemo<Position[]>(() => {
    if (!isActionPhase) return []
    if (placingCatastrophe) {
      // Highlight all cells where catastrophe can be placed
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
      return getValidLeaderPlacements(state, selectedLeader)
    }
    return []
  }, [state, selectedTile, selectedLeader, placingCatastrophe, isActionPhase])

  // Handle cell click on board
  const handleCellClick = useCallback((pos: Position) => {
    if (!isActionPhase) return

    if (placingCatastrophe) {
      dispatch({
        type: 'placeCatastrophe',
        position: pos,
        playerIndex: state.currentPlayer,
      })
      setPlacingCatastrophe(false)
      return
    }

    if (selectedTile) {
      dispatch({
        type: 'placeTile',
        color: selectedTile,
        position: pos,
        playerIndex: state.currentPlayer,
      })
      setSelectedTile(null)
      return
    }

    if (selectedLeader) {
      dispatch({
        type: 'placeLeader',
        color: selectedLeader,
        position: pos,
        playerIndex: state.currentPlayer,
      })
      setSelectedLeader(null)
      return
    }
  }, [isActionPhase, placingCatastrophe, selectedTile, selectedLeader, state.currentPlayer, dispatch, setSelectedTile, setSelectedLeader, setPlacingCatastrophe])

  const handleSwapTiles = useCallback(() => {
    const indices = currentPlayer.hand.map((_, i) => i)
    dispatch({
      type: 'swapTiles',
      indices,
      playerIndex: state.currentPlayer,
    })
  }, [currentPlayer.hand, state.currentPlayer, dispatch])

  const handlePass = useCallback(() => {
    dispatch({
      type: 'pass',
      playerIndex: state.currentPlayer,
    })
  }, [state.currentPlayer, dispatch])

  const handleCommitSupport = useCallback((indices: number[]) => {
    // Determine who is committing
    const conflict = state.pendingConflict
    if (!conflict) return
    const committingPlayer = conflict.attackerCommitted === null
      ? conflict.attacker.playerIndex
      : conflict.defender.playerIndex
    dispatch({
      type: 'commitSupport',
      indices,
      playerIndex: committingPlayer,
    })
  }, [state.pendingConflict, dispatch])

  const handleBuildMonument = useCallback((monumentId: string) => {
    dispatch({
      type: 'buildMonument',
      monumentId,
      playerIndex: state.currentPlayer,
    })
  }, [state.currentPlayer, dispatch])

  const handleDeclineMonument = useCallback(() => {
    dispatch({
      type: 'declineMonument',
      playerIndex: state.currentPlayer,
    })
  }, [state.currentPlayer, dispatch])

  const handleChooseWarOrder = useCallback((color: LeaderColor) => {
    dispatch({
      type: 'chooseWarOrder',
      color,
      playerIndex: state.currentPlayer,
    })
  }, [state.currentPlayer, dispatch])

  const handleWithdrawLeader = useCallback((color: LeaderColor) => {
    dispatch({
      type: 'withdrawLeader',
      color,
      playerIndex: state.currentPlayer,
    })
  }, [state.currentPlayer, dispatch])

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: '#1a1a2e',
      overflow: 'hidden',
    }}>
      <TopBar state={state} />

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'auto',
          padding: '16px',
        }}>
          <GameBoard
            state={state}
            highlights={highlights}
            onCellClick={handleCellClick}
          />
        </div>
        <PlayerPanel state={state} />
      </div>

      <HandPanel
        hand={currentPlayer.hand}
        selectedTile={selectedTile}
        onSelectTile={setSelectedTile}
        disabled={!isActionPhase}
      />

      <ActionBar
        state={state}
        selectedLeader={selectedLeader}
        placingCatastrophe={placingCatastrophe}
        onSelectLeader={setSelectedLeader}
        onPlaceCatastrophe={setPlacingCatastrophe}
        onSwapTiles={handleSwapTiles}
        onPass={handlePass}
        onWithdrawLeader={handleWithdrawLeader}
        disabled={!isActionPhase}
      />

      {/* Dialog overlays */}
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

      {/* New Game button (temporary, for dev) */}
      <div style={{
        position: 'fixed',
        top: '8px',
        right: '8px',
        zIndex: 100,
      }}>
        <button
          onClick={() => startNewGame(2)}
          style={{
            padding: '4px 10px',
            borderRadius: '4px',
            border: '1px solid #0f3460',
            background: '#0f1b3e',
            color: '#888',
            cursor: 'pointer',
            fontSize: '11px',
            fontFamily: 'sans-serif',
          }}
        >
          New Game
        </button>
      </div>
    </div>
  )
}

export default App
