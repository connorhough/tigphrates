import { runGame, runTournament } from './runGame'

const args = process.argv.slice(2)
const playerCount = parseInt(args.find(a => a.startsWith('--players='))?.split('=')[1] ?? '2', 10)
const gameCount = parseInt(args.find(a => a.startsWith('--games='))?.split('=')[1] ?? '1', 10)
const maxTurns = args.find(a => a.startsWith('--turns='))
  ? parseInt(args.find(a => a.startsWith('--turns='))!.split('=')[1], 10)
  : undefined
const showLog = args.includes('--log')
const noLog = args.includes('--no-log')

if (playerCount < 2 || playerCount > 4) {
  console.error('Player count must be 2-4')
  process.exit(1)
}

if (gameCount < 1) {
  console.error('Game count must be at least 1')
  process.exit(1)
}

if (gameCount === 1) {
  // Default: show log for single games unless --no-log
  const log = !noLog
  const label = maxTurns ? `${maxTurns} turns` : 'full game'
  console.log(`Running 1 AI game (${label}) with ${playerCount} players...\n`)
  const result = runGame({ playerCount, maxTurns, log })

  if (log && result.log.length > 0) {
    for (const line of result.log) console.log(line)
    console.log()
  }

  console.log(`Game finished (${result.reason}), ${result.turns} turns, ${result.turnCount} actions\n`)
  console.log('Results:')
  for (let i = 0; i < result.scores.length; i++) {
    const s = result.scores[i]
    const marker = i === result.winner ? ' *** WINNER ***' : ''
    console.log(
      `  Player ${i + 1} (${s.dynasty}): min=${s.minScore} | R=${s.score.red} B=${s.score.blue} G=${s.score.green} K=${s.score.black} | treasures=${s.treasures}${marker}`,
    )
  }
} else {
  // Default: no log for tournaments unless --log
  const log = showLog
  console.log(`Running ${gameCount} AI games with ${playerCount} players...\n`)
  const { results, wins, avgMinScores } = runTournament(gameCount, playerCount)

  if (log) {
    for (let g = 0; g < results.length; g++) {
      console.log(`\n--- Game ${g + 1} ---`)
      for (const line of results[g].log) console.log(line)
    }
    console.log()
  }

  console.log('Tournament Results:')
  for (let i = 0; i < playerCount; i++) {
    console.log(
      `  Player ${i + 1} (${results[0].scores[i].dynasty}): wins=${wins[i]}/${gameCount} (${((wins[i] / gameCount) * 100).toFixed(1)}%) | avg min score=${avgMinScores[i].toFixed(1)}`,
    )
  }

  console.log('\nIndividual games:')
  for (let g = 0; g < results.length; g++) {
    const r = results[g]
    const scores = r.scores.map((s, i) =>
      `P${i + 1}:${s.minScore}${i === r.winner ? '*' : ''}`,
    ).join(' ')
    console.log(`  Game ${g + 1}: ${scores} (${r.reason}, ${r.turns}t/${r.turnCount}a)`)
  }
}
