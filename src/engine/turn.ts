import { GameState, TileColor } from './types'
import { findKingdoms } from './board'
import { scoreMonuments } from './monument'

const STARTING_TEMPLES: [number, number][] = [
  [0,10],[1,1],[1,15],[2,5],[4,13],[6,9],[7,1],[8,14],[9,6],[10,10],
]
const STARTING_TEMPLE_SET = new Set(STARTING_TEMPLES.map(([r, c]) => `${r},${c}`))

/**
 * Collect treasures from kingdoms that have a green leader (trader) and 2+ treasures.
 * Removes treasures until only 1 remains per kingdom. Starting temple treasures are collected first.
 * Mutates state.
 */
export function collectTreasures(state: GameState): GameState {
  const kingdoms = findKingdoms(state.board)

  for (const kingdom of kingdoms) {
    // Check for a green leader (trader)
    const trader = kingdom.leaders.find(l => l.color === 'green')
    if (!trader) continue

    // Find cells with treasures in this kingdom
    const treasureCells = kingdom.positions.filter(
      p => state.board[p.row][p.col].hasTreasure
    )
    if (treasureCells.length < 2) continue

    // Find the player who owns the trader
    const owner = state.players.find(p =>
      p.dynasty === trader.dynasty
    )
    if (!owner) continue

    // Sort: starting temple treasures first (collected first)
    treasureCells.sort((a, b) => {
      const aIsStarting = STARTING_TEMPLE_SET.has(`${a.row},${a.col}`) ? 0 : 1
      const bIsStarting = STARTING_TEMPLE_SET.has(`${b.row},${b.col}`) ? 0 : 1
      return aIsStarting - bIsStarting
    })

    // Collect treasures until only 1 remains
    const toCollect = treasureCells.length - 1
    for (let i = 0; i < toCollect; i++) {
      const pos = treasureCells[i]
      state.board[pos.row][pos.col].hasTreasure = false
      owner.treasures += 1
    }
  }

  return state
}

/**
 * Count total treasures remaining on the board.
 */
function countBoardTreasures(state: GameState): number {
  let count = 0
  for (const row of state.board) {
    for (const cell of row) {
      if (cell.hasTreasure) count++
    }
  }
  return count
}

/**
 * Check if the game should end.
 * Game ends if:
 * - 2 or fewer treasure tokens remain on the board, OR
 * - The bag is empty and the current player cannot draw enough tiles to fill hand to 6
 */
export function checkGameEnd(state: GameState): boolean {
  if (countBoardTreasures(state) <= 2) return true

  const currentPlayer = state.players[state.currentPlayer]
  const tilesToDraw = 6 - currentPlayer.hand.length
  if (tilesToDraw > 0 && state.bag.length < tilesToDraw) return true

  return false
}

/**
 * Main end-of-turn processing. Mutates state (caller should clone if needed).
 *
 * 1. Draw tiles to refill hand to 6
 * 2. Monument scoring
 * 3. Treasure collection
 * 4. Game end check
 * 5. Advance turn (if game not over)
 */
export function endTurn(state: GameState): GameState {
  // 1. Draw tiles
  const player = state.players[state.currentPlayer]
  const tilesToDraw = Math.min(6 - player.hand.length, state.bag.length)
  if (tilesToDraw > 0) {
    const drawn = state.bag.splice(0, tilesToDraw)
    player.hand.push(...drawn)
  }

  // 2. Monument scoring
  scoreMonuments(state)

  // 3. Treasure collection
  collectTreasures(state)

  // 4. Game end check
  if (checkGameEnd(state)) {
    state.turnPhase = 'gameOver'
    return state
  }

  // 5. Advance turn
  state.currentPlayer = (state.currentPlayer + 1) % state.players.length
  state.actionsRemaining = 2
  state.turnPhase = 'action'

  return state
}

/**
 * Calculate final scores for all players.
 * Each player's score = minimum across 4 color scores after optimally assigning treasures.
 * Treasures are wild — each adds +1 to any color. Optimal: fill weakest colors first.
 * Returns sorted by score descending, with tiebreak on second-lowest, third, fourth.
 */
export function calculateFinalScores(
  state: GameState
): { playerIndex: number; finalScore: number; colorScores: Record<TileColor, number> }[] {
  const results = state.players.map((player, playerIndex) => {
    const colorScores: Record<TileColor, number> = { ...player.score }
    let remaining = player.treasures

    // Greedily assign treasures to the weakest color
    while (remaining > 0) {
      const colors: TileColor[] = ['red', 'blue', 'green', 'black']
      colors.sort((a, b) => colorScores[a] - colorScores[b])
      colorScores[colors[0]] += 1
      remaining--
    }

    const sorted = Object.values(colorScores).sort((a, b) => a - b)
    const finalScore = sorted[0]

    return { playerIndex, finalScore, colorScores, _sorted: sorted }
  })

  // Sort descending by score, tiebreak by second-lowest, third, fourth
  results.sort((a, b) => {
    for (let i = 0; i < 4; i++) {
      if (a._sorted[i] !== b._sorted[i]) return b._sorted[i] - a._sorted[i]
    }
    return 0
  })

  return results.map(({ _sorted, ...rest }) => rest)
}
