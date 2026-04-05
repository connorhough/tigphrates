import { createGame } from '../setup'
import { applyAction } from '../actions'
import { GameState } from '../types'

/**
 * Helper: set up two separate kingdoms that can be united by placing a tile.
 *
 * Kingdom A around temple (9,6):
 *   - Player 0's red leader at (9,5)
 *   - Tiles at (9,7), (9,8)
 *
 * Kingdom B around temple (10,10):
 *   - Player 1's red leader at (10,11)
 *   - Tile at (10,9) and (9,10)
 *
 * Gap at (9,9). Placing a tile there unites the kingdoms. All cells are land.
 */
function setupWarScenario(): GameState {
  const state = createGame(2)

  // Kingdom A: temple at (9,6) with player 0's red leader at (9,5)
  state.board[9][5].leader = { color: 'red', dynasty: 'archer' }
  state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 9, col: 5 }
  // Extend kingdom A: (9,7), (9,8)
  state.board[9][7].tile = 'green'
  state.board[9][8].tile = 'green'

  // Kingdom B: temple at (10,10) with player 1's red leader at (10,11)
  state.board[10][11].leader = { color: 'red', dynasty: 'bull' }
  state.players[1].leaders.find(l => l.color === 'red')!.position = { row: 10, col: 11 }
  // Extend kingdom B: (10,9) and (9,10)
  state.board[10][9].tile = 'green'
  state.board[9][10].tile = 'green'

  // Give players tiles
  state.players[0].hand = ['green', 'green', 'green', 'green', 'green', 'green']
  state.players[1].hand = ['green', 'green', 'green', 'green', 'green', 'green']

  return state
}

describe('war', () => {
  it('triggers war when tile unites two kingdoms with same-colored leaders', () => {
    const state = setupWarScenario()

    // Place green tile at (9,9) to unite kingdoms
    const result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })

    expect(result.turnPhase).toBe('conflictSupport')
    expect(result.pendingConflict).not.toBeNull()
    expect(result.pendingConflict!.type).toBe('war')
    expect(result.pendingConflict!.color).toBe('red')
    // Active player (0) is attacker
    expect(result.pendingConflict!.attacker.playerIndex).toBe(0)
    expect(result.pendingConflict!.defender.playerIndex).toBe(1)
    // actionsRemaining should NOT be decremented yet
    expect(result.actionsRemaining).toBe(2)
  })

  it('active player chooses war order when multiple conflicts', () => {
    const state = createGame(2)

    // Kingdom A around temple (9,6): player 0's red + green leaders
    state.board[9][5].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 9, col: 5 }
    state.board[9][7].leader = { color: 'green', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'green')!.position = { row: 9, col: 7 }
    state.board[9][8].tile = 'green'

    // Kingdom B around temple (10,10): player 1's red + green leaders
    state.board[10][11].leader = { color: 'red', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'red')!.position = { row: 10, col: 11 }
    state.board[9][10].leader = { color: 'green', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'green')!.position = { row: 9, col: 10 }
    state.board[10][9].tile = 'green'

    state.players[0].hand = ['green', 'green', 'green', 'green', 'green', 'green']

    // Place green tile at (9,9) to unite — both red and green leaders conflict
    const result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })

    expect(result.turnPhase).toBe('warOrderChoice')
    expect(result.pendingConflict).not.toBeNull()
    expect(result.pendingConflict!.pendingWarColors).toBeDefined()
    expect(result.pendingConflict!.pendingWarColors!.length).toBe(2)
    expect(result.pendingConflict!.pendingWarColors).toContain('red')
    expect(result.pendingConflict!.pendingWarColors).toContain('green')
  })

  it('chooseWarOrder selects which war to resolve first', () => {
    const state = createGame(2)

    // Kingdom A around temple (9,6): player 0's red + green leaders
    state.board[9][5].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 9, col: 5 }
    state.board[9][7].leader = { color: 'green', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'green')!.position = { row: 9, col: 7 }
    state.board[9][8].tile = 'green'

    // Kingdom B around temple (10,10): player 1's red + green leaders
    state.board[10][11].leader = { color: 'red', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'red')!.position = { row: 10, col: 11 }
    state.board[9][10].leader = { color: 'green', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'green')!.position = { row: 9, col: 10 }
    state.board[10][9].tile = 'green'

    state.players[0].hand = ['green', 'green', 'green', 'green', 'green', 'green']
    state.players[1].hand = ['green', 'green', 'green', 'green', 'green', 'green']

    let result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })
    expect(result.turnPhase).toBe('warOrderChoice')

    // Choose red war first
    result = applyAction(result, { type: 'chooseWarOrder', color: 'red' })
    expect(result.turnPhase).toBe('conflictSupport')
    expect(result.pendingConflict!.color).toBe('red')
    expect(result.pendingConflict!.pendingWarColors).toEqual(['green'])
  })

  it('winner scores 1 per removed tile + 1 for leader', () => {
    const state = setupWarScenario()

    // Add more red tiles to the defender's side (kingdom B)
    // Kingdom B: temple at (10,10) [red], tiles at (10,9) [green], (9,10) [green], leader at (10,11)
    // Add red tiles clearly on defender's side only
    state.board[10][12].tile = 'red'  // connected to kingdom B via (10,11) leader
    state.board[9][11].tile = 'red'   // connected via (10,11) or (9,10)

    // Player 0 has red support tiles
    state.players[0].hand = ['red', 'red', 'red', 'red', 'red', 'green']
    state.players[1].hand = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue']

    // Place green tile at (9,9) to unite → red war
    let result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })
    expect(result.pendingConflict!.type).toBe('war')
    expect(result.pendingConflict!.color).toBe('red')

    // Attacker commits lots of red support
    result = applyAction(result, { type: 'commitSupport', indices: [0, 1, 2] })
    // Defender commits 0
    result = applyAction(result, { type: 'commitSupport', indices: [] })

    // Attacker wins. Defender side red tiles: (10,10) temple + (10,12) + (9,11) = 3 red tiles
    // Winner scores removed_tiles + 1 = 3 + 1 = 4 red VP
    expect(result.players[0].score.red).toBe(4)
  })

  it('loser tiles removed from game', () => {
    const state = setupWarScenario()

    // Add red tiles to defender side (clearly on B side only)
    state.board[10][12].tile = 'red'
    state.board[9][11].tile = 'red'

    // Attacker needs enough support to overcome defender's 3 red base tiles
    // Attacker base: 1 (temple 9,6), Defender base: 3 (temple 10,10 + 10,12 + 9,11)
    // Attacker needs > 3, so commit 3 red → total 4 > 3
    state.players[0].hand = ['red', 'red', 'red', 'red', 'red', 'green']
    state.players[1].hand = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue']

    let result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })

    // Attacker commits 3 red support tiles
    result = applyAction(result, { type: 'commitSupport', indices: [0, 1, 2] })
    result = applyAction(result, { type: 'commitSupport', indices: [] })

    // Defender's red leader should be gone
    expect(result.board[10][11].leader).toBeNull()
    expect(result.players[1].leaders.find(l => l.color === 'red')!.position).toBeNull()

    // Red tiles on defender side should be removed
    expect(result.board[10][10].tile).toBeNull()  // temple was red
    expect(result.board[10][12].tile).toBeNull()
    expect(result.board[9][11].tile).toBeNull()
  })

  it('unification tile does not count as supporter', () => {
    const state = setupWarScenario()

    // Change bridge tiles to red so there are red tiles on attacker side
    state.board[9][7].tile = 'red'
    state.board[9][8].tile = 'red'
    // And defender side
    state.board[9][10].tile = 'red'

    // Use red as the unification tile at (9,9)
    state.players[0].hand = ['red', 'red', 'red', 'red', 'red', 'red']
    state.players[1].hand = ['red', 'red', 'red', 'red', 'red', 'red']

    // Place red tile at (9,9) — this is the unification tile and also red (war color)
    const result = applyAction(state, { type: 'placeTile', color: 'red', position: { row: 9, col: 9 } })

    expect(result.pendingConflict!.type).toBe('war')
    expect(result.pendingConflict!.color).toBe('red')

    // Attacker side red tiles: (9,6) temple, (9,7), (9,8) = 3
    // Defender side red tiles: (10,10) temple, (9,10) = 2
    // (9,9) the unification tile should NOT be counted for either side
    expect(result.pendingConflict!.attackerStrength).toBe(3)
    expect(result.pendingConflict!.defenderStrength).toBe(2)
  })

  it('defender wins ties', () => {
    const state = setupWarScenario()

    // Both sides: 1 red tile each (the temples)
    // Attacker side: temple at (9,6) = 1 red
    // Defender side: temple at (10,10) = 1 red
    state.players[0].hand = ['red', 'green', 'green', 'green', 'green', 'green']
    state.players[1].hand = ['red', 'green', 'green', 'green', 'green', 'green']

    let result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })

    expect(result.pendingConflict!.attackerStrength).toBe(1)
    expect(result.pendingConflict!.defenderStrength).toBe(1)

    // Both commit 1 red tile → tie
    result = applyAction(result, { type: 'commitSupport', indices: [0] })
    result = applyAction(result, { type: 'commitSupport', indices: [0] })

    // Tie: defender wins. Attacker's leader removed.
    expect(result.board[9][5].leader).toBeNull()
    expect(result.players[0].leaders.find(l => l.color === 'red')!.position).toBeNull()
    // Defender's leader stays
    expect(result.board[10][11].leader).toEqual({ color: 'red', dynasty: 'bull' })
    // Defender scores: removed red tiles on attacker side (temple at (9,6) = 1) + 1 = 2
    expect(result.players[1].score.red).toBe(2)
  })

  it('leaders without temple adjacency withdrawn after war', () => {
    const state = createGame(2)

    // Kingdom A around temple (9,6):
    // Player 0's red leader at (9,5), plus a blue leader at (9,7)
    // Blue leader at (9,7) is adjacent to (9,6) which is a red temple.
    // But also add red tile at (9,8) near blue leader.
    state.board[9][5].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 9, col: 5 }
    // Extra red tile at (10,6) to give attacker base
    state.board[10][6].tile = 'red'
    // Blue leader at (10,5) — adjacent to (10,6) [red] and (9,5) [leader]
    // Actually put the blue leader adjacent ONLY to the red temple.
    // (9,7) is adjacent to (9,6) [temple], so blue leader there.
    // After war, if attacker loses, red tiles on attacker side removed: (9,6) temple.
    // Then blue leader at (9,7) no longer adjacent to any red tile → withdrawn.
    state.board[9][7].leader = { color: 'blue', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'blue')!.position = { row: 9, col: 7 }
    state.board[9][8].tile = 'green'

    // Kingdom B around temple (10,10):
    state.board[10][11].leader = { color: 'red', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'red')!.position = { row: 10, col: 11 }
    state.board[10][9].tile = 'green'
    state.board[9][10].tile = 'green'
    // Add red tiles to make defender strong
    state.board[10][12].tile = 'red'

    // Give defender red support, attacker none
    state.players[0].hand = ['green', 'green', 'green', 'green', 'green', 'green']
    state.players[1].hand = ['red', 'red', 'red', 'red', 'red', 'red']

    // Place tile to unite
    let result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })

    expect(result.pendingConflict!.color).toBe('red')

    // Attacker commits 0 (no red tiles)
    result = applyAction(result, { type: 'commitSupport', indices: [] })
    // Defender commits 3 red tiles
    result = applyAction(result, { type: 'commitSupport', indices: [0, 1, 2] })

    // Defender wins: attacker's red leader removed + red tiles on attacker side removed
    expect(result.board[9][5].leader).toBeNull()
    // Red tile at (9,6) temple removed, (10,6) red removed
    expect(result.board[9][6].tile).toBeNull()
    expect(result.board[10][6].tile).toBeNull()

    // Blue leader at (9,7): neighbors are (9,6) [now empty], (9,8) [green tile], (8,7) [empty]
    // No adjacent face-up red temple → withdrawn
    expect(result.board[9][7].leader).toBeNull()
    expect(result.players[0].leaders.find(l => l.color === 'blue')!.position).toBeNull()
  })

  it('war support must match conflict color (not red for non-red wars)', () => {
    const state = createGame(2)

    // Set up green war: both kingdoms have green leaders
    // Kingdom A around temple (9,6): player 0's green leader at (9,5)
    state.board[9][5].leader = { color: 'green', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'green')!.position = { row: 9, col: 5 }
    state.board[9][7].tile = 'green'
    state.board[9][8].tile = 'green'

    // Kingdom B around temple (10,10): player 1's green leader at (10,11)
    state.board[10][11].leader = { color: 'green', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'green')!.position = { row: 10, col: 11 }
    state.board[10][9].tile = 'green'
    state.board[9][10].tile = 'green'

    state.players[0].hand = ['red', 'green', 'green', 'green', 'green', 'green']

    let result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })
    expect(result.pendingConflict!.type).toBe('war')
    expect(result.pendingConflict!.color).toBe('green')

    // Try to commit a red tile — should fail for green war
    expect(() => applyAction(result, { type: 'commitSupport', indices: [0] })).toThrow()
  })

  it('does not trigger war if tile unites regions without leaders', () => {
    const state = createGame(2)

    // Two regions (no leaders) connected by a tile
    state.board[5][5].tile = 'green'
    state.board[5][7].tile = 'green'
    state.players[0].hand = ['green', 'green', 'green', 'green', 'green', 'green']

    const result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 5, col: 6 } })
    expect(result.turnPhase).toBe('action')
    expect(result.pendingConflict).toBeNull()
    expect(result.actionsRemaining).toBe(1)
  })

  it('does not trigger war if kingdoms share no leader colors', () => {
    const state = createGame(2)

    // Kingdom A around temple (9,6): player 0's red leader
    state.board[9][5].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 9, col: 5 }
    state.board[9][7].tile = 'green'
    state.board[9][8].tile = 'green'

    // Kingdom B around temple (10,10): player 1's green leader (different color!)
    state.board[10][11].leader = { color: 'green', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'green')!.position = { row: 10, col: 11 }
    state.board[10][9].tile = 'green'
    state.board[9][10].tile = 'green'

    state.players[0].hand = ['green', 'green', 'green', 'green', 'green', 'green']

    const result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })
    // No war, just normal scoring
    expect(result.turnPhase).toBe('action')
    expect(result.pendingConflict).toBeNull()
    expect(result.actionsRemaining).toBe(1)
  })

  it('after war resolution returns to action phase with decremented actions', () => {
    const state = setupWarScenario()

    state.players[0].hand = ['red', 'red', 'red', 'green', 'green', 'green']
    state.players[1].hand = ['green', 'green', 'green', 'green', 'green', 'green']

    let result = applyAction(state, { type: 'placeTile', color: 'green', position: { row: 9, col: 9 } })
    // Attacker commits red tiles
    result = applyAction(result, { type: 'commitSupport', indices: [0, 1] })
    // Defender commits nothing
    result = applyAction(result, { type: 'commitSupport', indices: [] })

    expect(result.turnPhase).toBe('action')
    expect(result.actionsRemaining).toBe(1)
  })
})
