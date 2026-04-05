import { runGame, runTournament } from './runGame'

const args = process.argv.slice(2)
const playerCount = parseInt(args.find(a => a.startsWith('--players='))?.split('=')[1] ?? '2', 10)
const gameCount = parseInt(args.find(a => a.startsWith('--games='))?.split('=')[1] ?? '1', 10)

if (playerCount < 2 || playerCount > 4) {
  console.error('Player count must be 2-4')
  process.exit(1)
}

if (gameCount < 1) {
  console.error('Game count must be at least 1')
  process.exit(1)
}

if (gameCount === 1) {
  console.log(`Running 1 AI game with ${playerCount} players...\n`)
  const result = runGame(playerCount)

  console.log(`Game finished (${result.reason}), ${result.turnCount} actions taken\n`)
  console.log('Results:')
  for (let i = 0; i < result.scores.length; i++) {
    const s = result.scores[i]
    const marker = i === result.winner ? ' *** WINNER ***' : ''
    console.log(
      `  Player ${i + 1} (${s.dynasty}): min=${s.minScore} | R=${s.score.red} B=${s.score.blue} G=${s.score.green} K=${s.score.black} | treasures=${s.treasures}${marker}`,
    )
  }
} else {
  console.log(`Running ${gameCount} AI games with ${playerCount} players...\n`)
  const { results, wins, avgMinScores } = runTournament(gameCount, playerCount)

  console.log('Tournament Results:')
  for (let i = 0; i < playerCount; i++) {
    console.log(
      `  Player ${i + 1} (${results[0].scores[i].dynasty}): wins=${wins[i]}/${gameCount} (${((wins[i] / gameCount) * 100).toFixed(1)}%) | avg min score=${avgMinScores[i].toFixed(1)}`,
    )
  }

  // Show individual game results
  console.log('\nIndividual games:')
  for (let g = 0; g < results.length; g++) {
    const r = results[g]
    const scores = r.scores.map((s, i) =>
      `P${i + 1}:${s.minScore}${i === r.winner ? '*' : ''}`,
    ).join(' ')
    console.log(`  Game ${g + 1}: ${scores} (${r.reason}, ${r.turnCount} actions)`)
  }
}
