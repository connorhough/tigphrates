/**
 * Shared compact log formatting for game playback.
 * - Used by the headless runner (src/headless/runGame.ts) for tournament logs.
 * - Used by the browser (src/hooks/useGame.ts) to record live games and
 *   ship them to the lab server for review.
 */

import { GameState, GameAction, TileColor } from './types'

export const COLOR_GLYPH: Record<string, string> = {
  red: 'R', blue: 'B', green: 'G', black: 'K',
}

export function pos(r: number, c: number): string {
  return `${r}.${c}`
}

export function handSummary(hand: TileColor[]): string {
  const counts: Record<string, number> = {}
  for (const t of hand) counts[COLOR_GLYPH[t]] = (counts[COLOR_GLYPH[t]] || 0) + 1
  return Object.entries(counts).map(([c, n]) => `${n}${c}`).join('') || '(empty)'
}

export function scoreDelta(
  before: Record<TileColor, number>,
  after: Record<TileColor, number>,
): string {
  const parts: string[] = []
  for (const color of ['red', 'blue', 'green', 'black'] as TileColor[]) {
    const diff = after[color] - before[color]
    if (diff > 0) parts.push(`+${diff}${COLOR_GLYPH[color]}`)
  }
  return parts.join(' ')
}

export function treasureDelta(before: number, after: number): string {
  const diff = after - before
  return diff > 0 ? ` +${diff}tr` : ''
}

export function scoreSnapshot(players: GameState['players']): string {
  return players.map((p, i) =>
    `P${i + 1}[${COLOR_GLYPH.red}${p.score.red} ${COLOR_GLYPH.blue}${p.score.blue} ${COLOR_GLYPH.green}${p.score.green} ${COLOR_GLYPH.black}${p.score.black} tr${p.treasures}]`,
  ).join(' ')
}

export function formatAction(action: GameAction): string {
  switch (action.type) {
    case 'placeTile': return `T:${COLOR_GLYPH[action.color]}@${pos(action.position.row, action.position.col)}`
    case 'placeLeader': return `L:${COLOR_GLYPH[action.color]}@${pos(action.position.row, action.position.col)}`
    case 'withdrawLeader': return `W:${COLOR_GLYPH[action.color]}`
    case 'placeCatastrophe': return `C@${pos(action.position.row, action.position.col)}`
    case 'swapTiles': return `S(${action.indices.length})`
    case 'pass': return 'P'
    case 'commitSupport': return `CS(${action.indices.length})`
    case 'chooseWarOrder': return `WO:${COLOR_GLYPH[action.color]}`
    case 'buildMonument': return `BM:${action.monumentId}`
    case 'declineMonument': return 'DM'
    default: return '??'
  }
}

/**
 * Returns a single line describing a conflict resolution if `prev` had a
 * pending conflict and `next` cleared it. Used by the live log to surface
 * REVOLT / WAR outcomes after the dispatch that finalized them.
 */
export function conflictResolutionLine(prev: GameState, next: GameState): string | null {
  const pc = prev.pendingConflict
  if (!pc) return null
  if (next.pendingConflict === pc) return null
  if (pc.attackerCommitted === null || pc.defenderCommitted === null) return null
  const kind = pc.type === 'revolt' ? 'REVOLT' : 'WAR'
  const atkTotal = pc.attackerStrength + pc.attackerCommitted.length
  const defTotal = pc.defenderStrength + pc.defenderCommitted.length
  const winner = atkTotal > defTotal ? 'atk' : 'def'
  return `  ${kind}(${COLOR_GLYPH[pc.color]}) P${pc.attacker.playerIndex + 1}(${atkTotal}) vs P${pc.defender.playerIndex + 1}(${defTotal}) -> ${winner} wins`
}

/**
 * Build a single live-action log line from an (action, prev, next, player)
 * tuple. Mirrors the format runGame.ts uses for headless tournaments so a
 * single parser can read both.
 */
export function formatActionLine(
  action: GameAction,
  prev: GameState,
  next: GameState,
  fallback = false,
): string {
  const actionStr = fallback ? `${formatAction(action)}!fb` : formatAction(action)
  const deltas: string[] = []
  for (let i = 0; i < prev.players.length; i++) {
    const sd = scoreDelta(prev.players[i].score, next.players[i].score)
    const td = treasureDelta(prev.players[i].treasures, next.players[i].treasures)
    if (sd || td) deltas.push(`P${i + 1}:${sd}${td}`)
  }
  const deltaStr = deltas.length ? ` [${deltas.join(', ')}]` : ''
  let phaseStr = ''
  if (next.turnPhase !== prev.turnPhase && next.turnPhase !== 'action') {
    phaseStr = ` ->${next.turnPhase}`
  }
  return `  ${actionStr}${deltaStr}${phaseStr}`
}
