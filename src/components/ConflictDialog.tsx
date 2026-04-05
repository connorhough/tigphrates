import { useState, useCallback } from 'react'
import { GameState, TileColor } from '../engine/types'

interface ConflictDialogProps {
  state: GameState
  currentViewingPlayer: number
  onCommitSupport: (indices: number[]) => void
}

const COLOR_HEX: Record<string, string> = {
  red: '#e74c3c',
  blue: '#3498db',
  green: '#2ecc71',
  black: '#888',
}

const COLOR_LABELS: Record<string, string> = {
  red: 'Temple (Priest)',
  blue: 'Farm (Farmer)',
  green: 'Market (Trader)',
  black: 'Settlement (King)',
}

export function ConflictDialog({ state, currentViewingPlayer, onCommitSupport }: ConflictDialogProps) {
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set())

  const conflict = state.pendingConflict
  if (!conflict) return null

  // Determine who needs to commit
  const attackerNeedsToCommit = conflict.attackerCommitted === null
  const defenderNeedsToCommit = conflict.attackerCommitted !== null && conflict.defenderCommitted === null
  const committingPlayerIndex = attackerNeedsToCommit
    ? conflict.attacker.playerIndex
    : conflict.defender.playerIndex

  const isMyTurnToCommit = committingPlayerIndex === currentViewingPlayer

  // Determine which color of tiles can be committed
  const commitColor: TileColor = conflict.type === 'revolt' ? 'red' : conflict.color

  // Get the committing player's hand, filter for relevant tiles
  const hand = state.players[committingPlayerIndex]?.hand ?? []
  const relevantTileIndices = hand
    .map((color, idx) => ({ color, idx }))
    .filter(t => t.color === commitColor)

  const toggleIndex = useCallback((idx: number) => {
    setSelectedIndices(prev => {
      const next = new Set(prev)
      if (next.has(idx)) {
        next.delete(idx)
      } else {
        next.add(idx)
      }
      return next
    })
  }, [])

  const handleCommit = useCallback(() => {
    onCommitSupport(Array.from(selectedIndices))
    setSelectedIndices(new Set())
  }, [selectedIndices, onCommitSupport])

  const attackerDynasty = state.players[conflict.attacker.playerIndex]?.dynasty ?? '?'
  const defenderDynasty = state.players[conflict.defender.playerIndex]?.dynasty ?? '?'

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'rgba(0,0,0,0.7)',
      zIndex: 1000,
    }}>
      <div style={{
        background: '#1a1a2e',
        border: '2px solid #0f3460',
        borderRadius: '12px',
        padding: '24px 32px',
        minWidth: '360px',
        maxWidth: '480px',
        fontFamily: 'sans-serif',
        color: '#e0e0e0',
      }}>
        {/* Header */}
        <div style={{
          fontSize: '18px',
          fontWeight: 'bold',
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <span style={{
            display: 'inline-block',
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: COLOR_HEX[conflict.color],
          }} />
          {conflict.type === 'revolt' ? 'Revolt' : 'War'} — {COLOR_LABELS[conflict.color]}
        </div>

        {/* Attacker vs Defender */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: '16px',
          gap: '16px',
        }}>
          <div style={{
            flex: 1,
            background: '#0f1b3e',
            borderRadius: '8px',
            padding: '12px',
            border: attackerNeedsToCommit ? '1px solid #f39c12' : '1px solid #0f3460',
          }}>
            <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>ATTACKER</div>
            <div style={{ fontSize: '14px', fontWeight: 'bold', textTransform: 'capitalize' }}>
              {attackerDynasty} (P{conflict.attacker.playerIndex + 1})
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLOR_HEX[conflict.color], marginTop: '4px' }}>
              {conflict.attackerStrength}
              {conflict.attackerCommitted !== null && (
                <span style={{ fontSize: '12px', color: '#888', marginLeft: '6px' }}>
                  +{conflict.attackerCommitted.length} committed
                </span>
              )}
            </div>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            fontSize: '20px',
            color: '#555',
            fontWeight: 'bold',
          }}>
            vs
          </div>

          <div style={{
            flex: 1,
            background: '#0f1b3e',
            borderRadius: '8px',
            padding: '12px',
            border: defenderNeedsToCommit ? '1px solid #f39c12' : '1px solid #0f3460',
          }}>
            <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>DEFENDER</div>
            <div style={{ fontSize: '14px', fontWeight: 'bold', textTransform: 'capitalize' }}>
              {defenderDynasty} (P{conflict.defender.playerIndex + 1})
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLOR_HEX[conflict.color], marginTop: '4px' }}>
              {conflict.defenderStrength}
              {conflict.defenderCommitted !== null && (
                <span style={{ fontSize: '12px', color: '#888', marginLeft: '6px' }}>
                  +{conflict.defenderCommitted.length} committed
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Commit area or waiting */}
        {isMyTurnToCommit ? (
          <div>
            <div style={{ fontSize: '13px', color: '#aaa', marginBottom: '8px' }}>
              Select {commitColor} tiles to commit as support ({relevantTileIndices.length} available):
            </div>

            {relevantTileIndices.length === 0 ? (
              <div style={{ color: '#666', fontSize: '13px', marginBottom: '12px' }}>
                No matching tiles in hand.
              </div>
            ) : (
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                {relevantTileIndices.map(({ idx }) => {
                  const isSelected = selectedIndices.has(idx)
                  return (
                    <button
                      key={idx}
                      onClick={() => toggleIndex(idx)}
                      style={{
                        width: '36px',
                        height: '36px',
                        borderRadius: '6px',
                        border: isSelected ? '2px solid #fff' : '2px solid #333',
                        background: isSelected ? COLOR_HEX[commitColor] : '#0f1b3e',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '12px',
                        color: '#ccc',
                      }}
                    >
                      <div style={{
                        width: '18px',
                        height: '18px',
                        borderRadius: '3px',
                        background: COLOR_HEX[commitColor],
                        opacity: isSelected ? 1 : 0.5,
                      }} />
                    </button>
                  )
                })}
              </div>
            )}

            <button
              onClick={handleCommit}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '6px',
                border: '1px solid #0f3460',
                background: '#1a2a5e',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '14px',
                fontFamily: 'sans-serif',
                fontWeight: 'bold',
              }}
            >
              Commit Support ({selectedIndices.size} tile{selectedIndices.size !== 1 ? 's' : ''})
            </button>
          </div>
        ) : (
          <div style={{
            textAlign: 'center',
            padding: '16px',
            color: '#888',
            fontSize: '14px',
          }}>
            Waiting for {state.players[committingPlayerIndex]?.dynasty ?? 'opponent'} to commit support...
          </div>
        )}
      </div>
    </div>
  )
}
