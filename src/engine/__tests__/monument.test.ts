import { createGame } from '../setup'
import { applyAction } from '../actions'
import { find2x2Square, getAvailableMonuments, buildMonument, scoreMonuments } from '../monument'
import { findKingdoms } from '../board'

function placeRedTile(board: ReturnType<typeof createGame>['board'], row: number, col: number) {
  board[row][col].tile = 'red'
  board[row][col].tileFlipped = false
  board[row][col].terrain = 'land'
  board[row][col].monument = null
}

function clearCell(board: ReturnType<typeof createGame>['board'], row: number, col: number) {
  board[row][col].tile = null
  board[row][col].tileFlipped = false
  board[row][col].leader = null
  board[row][col].catastrophe = false
  board[row][col].monument = null
  board[row][col].hasTreasure = false
}

describe('monuments', () => {
  it('detects 2x2 same-color square after tile placement', () => {
    const state = createGame(2)
    // Place 4 red tiles in a 2x2 at rows 4-5, cols 4-5 (land area)
    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)
    placeRedTile(state.board, 5, 5)

    // Checking from the last placed position
    const result = find2x2Square(state.board, { row: 5, col: 5 })
    expect(result).not.toBeNull()
    expect(result!.topLeft).toEqual({ row: 4, col: 4 })
    expect(result!.color).toBe('red')
  })

  it('does not detect 2x2 if tiles are different colors', () => {
    const state = createGame(2)
    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)
    state.board[5][5].tile = 'green'
    state.board[5][5].terrain = 'land'
    state.board[5][5].tileFlipped = false

    const result = find2x2Square(state.board, { row: 5, col: 5 })
    expect(result).toBeNull()
  })

  it('does not detect 2x2 if a tile is flipped', () => {
    const state = createGame(2)
    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)
    placeRedTile(state.board, 5, 5)
    state.board[4][4].tileFlipped = true

    const result = find2x2Square(state.board, { row: 5, col: 5 })
    expect(result).toBeNull()
  })

  it('building flips tiles face-down and places monument', () => {
    const state = createGame(2)
    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)
    placeRedTile(state.board, 5, 5)

    const topLeft = { row: 4, col: 4 }
    buildMonument(state, 'red-blue', topLeft)

    // All 4 tiles should be flipped
    expect(state.board[4][4].tileFlipped).toBe(true)
    expect(state.board[4][5].tileFlipped).toBe(true)
    expect(state.board[5][4].tileFlipped).toBe(true)
    expect(state.board[5][5].tileFlipped).toBe(true)

    // All 4 cells should reference the monument
    expect(state.board[4][4].monument).toBe('red-blue')
    expect(state.board[4][5].monument).toBe('red-blue')
    expect(state.board[5][4].monument).toBe('red-blue')
    expect(state.board[5][5].monument).toBe('red-blue')

    // Monument position should be set
    const monument = state.monuments.find(m => m.id === 'red-blue')!
    expect(monument.position).toEqual(topLeft)
  })

  it('declining monument leaves tiles face-up', () => {
    const state = createGame(2)
    // Set up a 2x2 of red tiles
    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)

    // Give the player a red tile and place it
    state.players[0].hand[0] = 'red'
    // Place a leader adjacent to a temple so the tile scores to a kingdom
    // Actually, just use applyAction to place the 4th tile
    const result = applyAction(state, { type: 'placeTile', color: 'red', position: { row: 5, col: 5 } })

    // Should be in monument choice phase
    expect(result.turnPhase).toBe('monumentChoice')
    expect(result.pendingMonument).not.toBeNull()

    // Decline the monument
    const afterDecline = applyAction(result, { type: 'declineMonument' })

    // Tiles should still be face-up
    expect(afterDecline.board[4][4].tileFlipped).toBe(false)
    expect(afterDecline.board[4][5].tileFlipped).toBe(false)
    expect(afterDecline.board[5][4].tileFlipped).toBe(false)
    expect(afterDecline.board[5][5].tileFlipped).toBe(false)
    expect(afterDecline.turnPhase).toBe('action')
  })

  it('monument must have a matching color', () => {
    const state = createGame(2)
    const available = getAvailableMonuments(state.monuments, 'red')
    // red matches red-blue, red-green, red-black
    expect(available).toHaveLength(3)
    expect(available.every(m => m.color1 === 'red' || m.color2 === 'red')).toBe(true)

    // Already-placed monument should not be available
    state.monuments[0].position = { row: 0, col: 0 }
    const available2 = getAvailableMonuments(state.monuments, 'red')
    expect(available2).toHaveLength(2)
  })

  it('face-down tiles still connect but dont count as supporters', () => {
    const state = createGame(2)
    // The monument tiles are flipped but should still connect the two sides

    // Clear an area and set up tiles
    for (let r = 3; r <= 6; r++) {
      for (let c = 2; c <= 8; c++) {
        clearCell(state.board, r, c)
      }
    }

    // Place red tiles to form a connected group with a leader on each side
    placeRedTile(state.board, 5, 3) // temple for leader adjacency (left)
    placeRedTile(state.board, 5, 4) // connects to monument
    placeRedTile(state.board, 5, 5) // } monument 2x2
    placeRedTile(state.board, 5, 6) // }
    placeRedTile(state.board, 4, 5) // }
    placeRedTile(state.board, 4, 6) // }
    placeRedTile(state.board, 5, 7) // connects on other side
    placeRedTile(state.board, 4, 7) // temple for leader adjacency (right)

    // Place leaders on each side
    state.board[5][2].leader = { color: 'red', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 5, col: 2 }
    state.board[4][8].leader = { color: 'blue', dynasty: 'bull' }
    state.players[1].leaders.find(l => l.color === 'blue')!.position = { row: 4, col: 8 }

    // Build monument at (4,5)-(5,6)
    buildMonument(state, 'red-blue', { row: 4, col: 5 })

    // The flipped tiles should still connect both sides
    const kingdoms = findKingdoms(state.board)
    // Both leaders should be in the same kingdom (connected through flipped monument tiles)
    const kingdom = kingdoms.find(k =>
      k.leaders.some(l => l.color === 'red') &&
      k.leaders.some(l => l.color === 'blue')
    )
    expect(kingdom).toBeDefined()
  })

  it('leader withdrawn if monument flip removes temple adjacency', () => {
    const state = createGame(2)

    // Clear area
    for (let r = 3; r <= 6; r++) {
      for (let c = 3; c <= 7; c++) {
        clearCell(state.board, r, c)
      }
    }

    // Create a 2x2 of red tiles
    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)
    placeRedTile(state.board, 5, 5)

    // Place a leader at (4, 3), adjacent to the red tile at (4, 4)
    // This is the only adjacent temple for this leader
    state.board[4][3].leader = { color: 'blue', dynasty: 'archer' }
    state.players[0].leaders.find(l => l.color === 'blue')!.position = { row: 4, col: 3 }

    // Building the monument flips (4,4) face-down, removing temple adjacency
    buildMonument(state, 'red-blue', { row: 4, col: 4 })

    // Leader should be withdrawn
    expect(state.board[4][3].leader).toBeNull()
    expect(state.players[0].leaders.find(l => l.color === 'blue')!.position).toBeNull()
  })

  it('monument scoring awards VP per matching leader in kingdom', () => {
    const state = createGame(2)

    // Clear area
    for (let r = 3; r <= 7; r++) {
      for (let c = 3; c <= 8; c++) {
        clearCell(state.board, r, c)
      }
    }

    // Build a connected group with a monument and leaders
    // Place face-up red tile for leader adjacency
    placeRedTile(state.board, 5, 3) // temple for leader adjacency

    // Place monument tiles (already flipped by buildMonument)
    placeRedTile(state.board, 5, 5)
    placeRedTile(state.board, 5, 6)
    placeRedTile(state.board, 6, 5)
    placeRedTile(state.board, 6, 6)

    // Connect them
    placeRedTile(state.board, 5, 4) // bridge between leader area and monument

    // Build a red-blue monument
    buildMonument(state, 'red-blue', { row: 5, col: 5 })

    // Place a red leader (priest) in the kingdom, adjacent to the temple at (5,3)
    state.board[5][2] = { ...state.board[5][2], leader: { color: 'red', dynasty: 'archer' }, tile: null, terrain: 'land', tileFlipped: false, catastrophe: false, monument: null, hasTreasure: false }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 5, col: 2 }

    // Score
    const before = { ...state.players[0].score }
    scoreMonuments(state)

    // Player 0 has red leader in kingdom with red-blue monument => +1 red VP
    expect(state.players[0].score.red).toBe(before.red + 1)
    // No blue leader => no blue VP
    expect(state.players[0].score.blue).toBe(before.blue)
  })

  it('monument scoring awards both colors if player has both leaders', () => {
    const state = createGame(2)

    // Clear area
    for (let r = 3; r <= 7; r++) {
      for (let c = 2; c <= 9; c++) {
        clearCell(state.board, r, c)
      }
    }

    // Set up connected region with monument
    placeRedTile(state.board, 5, 3) // temple
    placeRedTile(state.board, 5, 4) // bridge
    placeRedTile(state.board, 5, 5)
    placeRedTile(state.board, 5, 6)
    placeRedTile(state.board, 6, 5)
    placeRedTile(state.board, 6, 6)

    buildMonument(state, 'red-blue', { row: 5, col: 5 })

    // Place red leader adjacent to temple
    state.board[5][2] = { ...state.board[5][2], leader: { color: 'red', dynasty: 'archer' }, tile: null, terrain: 'land', tileFlipped: false, catastrophe: false, monument: null, hasTreasure: false }
    state.players[0].leaders.find(l => l.color === 'red')!.position = { row: 5, col: 2 }

    // Place blue leader also in the kingdom, adjacent to temple at (5,3)
    state.board[4][3] = { ...state.board[4][3], leader: { color: 'blue', dynasty: 'archer' }, tile: null, terrain: 'land', tileFlipped: false, catastrophe: false, monument: null, hasTreasure: false }
    state.players[0].leaders.find(l => l.color === 'blue')!.position = { row: 4, col: 3 }

    const before = { ...state.players[0].score }
    scoreMonuments(state)

    expect(state.players[0].score.red).toBe(before.red + 1)
    expect(state.players[0].score.blue).toBe(before.blue + 1)
  })

  it('integrates with placeTile to trigger monumentChoice phase', () => {
    const state = createGame(2)

    // Set up 3 of 4 red tiles in a 2x2
    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)

    // Give player a red tile
    state.players[0].hand[0] = 'red'

    // Place the 4th tile
    const result = applyAction(state, { type: 'placeTile', color: 'red', position: { row: 5, col: 5 } })

    expect(result.turnPhase).toBe('monumentChoice')
    expect(result.pendingMonument).toEqual({ position: { row: 4, col: 4 }, color: 'red' })
    // Actions should NOT have been decremented yet
    expect(result.actionsRemaining).toBe(2)
  })

  it('buildMonument action clears pending and decrements actions', () => {
    const state = createGame(2)

    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)
    state.players[0].hand[0] = 'red'

    const afterPlace = applyAction(state, { type: 'placeTile', color: 'red', position: { row: 5, col: 5 } })
    expect(afterPlace.turnPhase).toBe('monumentChoice')

    const afterBuild = applyAction(afterPlace, { type: 'buildMonument', monumentId: 'red-blue' })
    expect(afterBuild.turnPhase).toBe('action')
    expect(afterBuild.pendingMonument).toBeNull()
    expect(afterBuild.actionsRemaining).toBe(1)
    expect(afterBuild.board[4][4].monument).toBe('red-blue')
  })

  it('no monumentChoice if no available monuments match', () => {
    const state = createGame(2)

    // Mark all red-matching monuments as already placed
    for (const m of state.monuments) {
      if (m.color1 === 'red' || m.color2 === 'red') {
        m.position = { row: 0, col: 0 }
      }
    }

    placeRedTile(state.board, 4, 4)
    placeRedTile(state.board, 4, 5)
    placeRedTile(state.board, 5, 4)
    state.players[0].hand[0] = 'red'

    const result = applyAction(state, { type: 'placeTile', color: 'red', position: { row: 5, col: 5 } })

    // Should NOT enter monument choice since no monuments available
    expect(result.turnPhase).toBe('action')
    expect(result.pendingMonument).toBeNull()
    expect(result.actionsRemaining).toBe(1)
  })
})
