import { useState, useCallback } from 'react'
import { GameState, TileColor } from '../engine/types'

interface ConflictDialogProps {
  state: GameState
  currentViewingPlayer: number
  onCommitSupport: (indices: number[]) => void
}

const TILE_VAR: Record<string, string> = {
  red: 'var(--tile-red)',
  blue: 'var(--tile-blue)',
  green: 'var(--tile-green)',
  black: 'var(--tile-black)',
}

const LABELS: Record<string, string> = {
  red: 'Temple (Priest)',
  blue: 'Farm (Farmer)',
  green: 'Market (Trader)',
  black: 'Settlement (King)',
}

export function ConflictDialog({
  state,
  currentViewingPlayer,
  onCommitSupport,
}: ConflictDialogProps) {
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const conflict = state.pendingConflict
  if (!conflict) return null

  const attackerTurn = conflict.attackerCommitted === null
  const defenderTurn =
    conflict.attackerCommitted !== null && conflict.defenderCommitted === null
  const committing = attackerTurn
    ? conflict.attacker.playerIndex
    : conflict.defender.playerIndex

  const isMyTurn = committing === currentViewingPlayer
  const commitColor: TileColor =
    conflict.type === 'revolt' ? 'red' : conflict.color
  const hand = state.players[committing]?.hand ?? []
  const relevant = hand
    .map((color, idx) => ({ color, idx }))
    .filter((t) => t.color === commitColor)

  const toggle = useCallback((idx: number) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }, [])

  const handleCommit = useCallback(() => {
    onCommitSupport(Array.from(selected))
    setSelected(new Set())
  }, [selected, onCommitSupport])

  const attackerDyn = state.players[conflict.attacker.playerIndex]?.dynasty ?? '?'
  const defenderDyn = state.players[conflict.defender.playerIndex]?.dynasty ?? '?'

  return (
    <Overlay>
      <Sheet>
        <Header>
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              background: TILE_VAR[conflict.color],
            }}
          />
          {conflict.type === 'revolt' ? 'Revolt' : 'War'} — {LABELS[conflict.color]}
        </Header>

        <div
          style={{
            display: 'flex',
            gap: 8,
            marginBottom: 12,
          }}
        >
          <Side
            label="Attacker"
            dynasty={attackerDyn}
            pi={conflict.attacker.playerIndex}
            strength={conflict.attackerStrength}
            committed={conflict.attackerCommitted?.length ?? null}
            color={TILE_VAR[conflict.color]}
            active={attackerTurn}
          />
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              color: 'var(--ink-faint)',
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              letterSpacing: 1,
            }}
          >
            vs
          </div>
          <Side
            label="Defender"
            dynasty={defenderDyn}
            pi={conflict.defender.playerIndex}
            strength={conflict.defenderStrength}
            committed={conflict.defenderCommitted?.length ?? null}
            color={TILE_VAR[conflict.color]}
            active={defenderTurn}
          />
        </div>

        {isMyTurn ? (
          <>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                color: 'var(--ink-faint)',
                textTransform: 'uppercase',
                letterSpacing: 1,
                marginBottom: 8,
              }}
            >
              Select {commitColor} tiles — {relevant.length} available
            </div>

            {relevant.length === 0 ? (
              <div
                style={{ color: 'var(--ink-faint)', fontSize: 13, marginBottom: 12 }}
              >
                No matching tiles in hand.
              </div>
            ) : (
              <div
                style={{
                  display: 'flex',
                  gap: 6,
                  flexWrap: 'wrap',
                  marginBottom: 12,
                  overflowY: 'auto',
                }}
              >
                {relevant.map(({ idx }) => {
                  const isSel = selected.has(idx)
                  return (
                    <button
                      key={idx}
                      onClick={() => toggle(idx)}
                      style={{
                        width: 44,
                        height: 44,
                        border: isSel
                          ? '2px solid var(--ink)'
                          : '1px solid var(--rule)',
                        background: isSel ? 'var(--paper-light)' : 'var(--paper)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <div
                        style={{
                          width: 22,
                          height: 22,
                          background: TILE_VAR[commitColor],
                          opacity: isSel ? 1 : 0.5,
                        }}
                      />
                    </button>
                  )
                })}
              </div>
            )}

            <PrimaryBtn onClick={handleCommit}>
              Commit Support ({selected.size} tile{selected.size !== 1 ? 's' : ''})
            </PrimaryBtn>
          </>
        ) : (
          <div
            style={{
              textAlign: 'center',
              padding: 16,
              color: 'var(--ink-faint)',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              letterSpacing: 1,
              textTransform: 'uppercase',
            }}
          >
            Waiting for {state.players[committing]?.dynasty ?? 'opponent'}…
          </div>
        )}
      </Sheet>
    </Overlay>
  )
}

function Side({
  label,
  dynasty,
  pi,
  strength,
  committed,
  color,
  active,
}: {
  label: string
  dynasty: string
  pi: number
  strength: number
  committed: number | null
  color: string
  active: boolean
}) {
  return (
    <div
      style={{
        flex: 1,
        background: 'var(--paper)',
        padding: 10,
        border: `1px solid ${active ? 'var(--dynasty-lion)' : 'var(--rule)'}`,
      }}
    >
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 9,
          color: 'var(--ink-faint)',
          letterSpacing: 1,
          textTransform: 'uppercase',
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontSize: 14,
          color: 'var(--ink)',
          textTransform: 'capitalize',
          marginBottom: 4,
        }}
      >
        {dynasty} · P{pi + 1}
      </div>
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontWeight: 700,
          fontSize: 20,
          color,
        }}
      >
        {strength}
        {committed !== null && (
          <span
            style={{
              fontSize: 11,
              color: 'var(--ink-faint)',
              marginLeft: 6,
            }}
          >
            +{committed}
          </span>
        )}
      </div>
    </div>
  )
}

export function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'center',
        background: 'rgba(0,0,0,0.6)',
        backdropFilter: 'blur(2px)',
        zIndex: 1000,
        paddingTop: 'var(--sat)',
      }}
    >
      {children}
    </div>
  )
}

export function Sheet({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        background: 'var(--paper)',
        border: '1px solid var(--rule)',
        padding: 20,
        paddingBottom: 'calc(20px + var(--sab))',
        width: '100%',
        maxWidth: 520,
        maxHeight: 'calc(100svh - var(--sat))',
        overflowY: 'auto',
        fontFamily: 'var(--font-body)',
        color: 'var(--ink)',
      }}
    >
      {children}
    </div>
  )
}

export function Header({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontFamily: 'var(--font-display)',
        fontWeight: 700,
        fontSize: 18,
        marginBottom: 14,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        color: 'var(--ink)',
      }}
    >
      {children}
    </div>
  )
}

export function PrimaryBtn({
  onClick,
  children,
}: {
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      style={{
        width: '100%',
        padding: 12,
        minHeight: 48,
        border: '1px solid var(--accent)',
        background: 'var(--paper-light)',
        color: 'var(--accent)',
        cursor: 'pointer',
        fontFamily: 'var(--font-mono)',
        fontSize: 12,
        fontWeight: 500,
        textTransform: 'uppercase',
        letterSpacing: 1,
      }}
    >
      {children}
    </button>
  )
}
