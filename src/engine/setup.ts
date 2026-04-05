import {
  GameState, Player, Dynasty, TileColor, LeaderColor,
  TILE_COUNTS, ALL_MONUMENTS, createInitialBoard
} from './types'

const DYNASTIES: Dynasty[] = ['archer', 'bull', 'pot', 'lion']
const LEADER_COLORS: LeaderColor[] = ['red', 'blue', 'green', 'black']

function createBag(): TileColor[] {
  const bag: TileColor[] = []
  for (const [color, count] of Object.entries(TILE_COUNTS)) {
    // Subtract 10 starting temples from red
    const adjusted = color === 'red' ? count - 10 : count
    for (let i = 0; i < adjusted; i++) {
      bag.push(color as TileColor)
    }
  }
  return shuffle(bag)
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function drawTiles(bag: TileColor[], count: number): { drawn: TileColor[]; remaining: TileColor[] } {
  return {
    drawn: bag.slice(0, count),
    remaining: bag.slice(count),
  }
}

export function createGame(
  playerCount: number,
  aiFlags?: boolean[],
): GameState {
  let bag = createBag()
  const players: Player[] = []

  for (let i = 0; i < playerCount; i++) {
    const { drawn, remaining } = drawTiles(bag, 6)
    bag = remaining
    players.push({
      dynasty: DYNASTIES[i],
      hand: drawn,
      leaders: LEADER_COLORS.map(color => ({ color, position: null })),
      catastrophesRemaining: 2,
      score: { red: 0, blue: 0, green: 0, black: 0 },
      treasures: 0,
      isAI: aiFlags?.[i] ?? (i > 0),
    })
  }

  return {
    board: createInitialBoard(),
    players,
    bag,
    monuments: ALL_MONUMENTS.map(m => ({ ...m })),
    currentPlayer: 0,
    actionsRemaining: 2,
    turnPhase: 'action',
    pendingConflict: null,
    pendingMonument: null,
  }
}
