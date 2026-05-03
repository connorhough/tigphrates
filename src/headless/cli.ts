import { runGame, runTournament, AiKind } from './runGame'

const args = process.argv.slice(2)
const playerCount = parseInt(args.find(a => a.startsWith('--players='))?.split('=')[1] ?? '2', 10)
const gameCount = parseInt(args.find(a => a.startsWith('--games='))?.split('=')[1] ?? '1', 10)
const maxTurns = args.find(a => a.startsWith('--turns='))
  ? parseInt(args.find(a => a.startsWith('--turns='))!.split('=')[1], 10)
  : undefined
const showLog = args.includes('--log')
const noLog = args.includes('--no-log')

// --ai-kind=onnx applies to player 1 (seat 0); other seats stay heuristic.
// --ai-kinds=onnx,simple,simple sets per-player.
// --model=<path> overrides the default ONNX path.
const aiKindFlag = args.find(a => a.startsWith('--ai-kind='))?.split('=')[1] as AiKind | undefined
const aiKindsFlag = args.find(a => a.startsWith('--ai-kinds='))?.split('=')[1]
const modelFlag = args.find(a => a.startsWith('--model='))?.split('=')[1]

let aiKinds: AiKind[] | undefined
if (aiKindsFlag) {
  aiKinds = aiKindsFlag.split(',').map(s => s.trim() as AiKind)
} else if (aiKindFlag) {
  // Single flag: seat 0 is onnx, rest are heuristic.
  aiKinds = new Array(playerCount).fill('simple') as AiKind[]
  aiKinds[0] = aiKindFlag
}

if (aiKinds && aiKinds.length !== playerCount) {
  console.error(`--ai-kinds count ${aiKinds.length} != --players=${playerCount}`)
  process.exit(1)
}
if (aiKinds && aiKinds.some(k => k !== 'simple' && k !== 'onnx')) {
  console.error(`--ai-kind values must be 'simple' or 'onnx'`)
  process.exit(1)
}

if (playerCount < 2 || playerCount > 4) {
  console.error('Player count must be 2-4')
  process.exit(1)
}

if (gameCount < 1) {
  console.error('Game count must be at least 1')
  process.exit(1)
}

async function main() {
  if (gameCount === 1) {
    const log = !noLog
    const label = maxTurns ? `${maxTurns} turns` : 'full game'
    const kindStr = aiKinds ? `[${aiKinds.join(',')}]` : '[all simple]'
    console.log(`Running 1 AI game (${label}) with ${playerCount} players ${kindStr}...\n`)
    const result = await runGame({ playerCount, maxTurns, log, aiKinds, onnxModelPath: modelFlag })

    if (log && result.log.length > 0) {
      for (const line of result.log) console.log(line)
      console.log()
    }

    console.log(`Game finished (${result.reason}), ${result.turns} turns, ${result.turnCount} actions\n`)
    console.log('Results:')
    for (let i = 0; i < result.scores.length; i++) {
      const s = result.scores[i]
      const marker = i === result.winner ? ' *** WINNER ***' : ''
      const kindLabel = aiKinds ? ` [${aiKinds[i]}]` : ''
      console.log(
        `  Player ${i + 1} (${s.dynasty})${kindLabel}: min=${s.minScore} | R=${s.score.red} B=${s.score.blue} G=${s.score.green} K=${s.score.black} | treasures=${s.treasures}${marker}`,
      )
    }
  } else {
    const log = showLog
    const kindStr = aiKinds ? `[${aiKinds.join(',')}]` : '[all simple]'
    console.log(`Running ${gameCount} AI games with ${playerCount} players ${kindStr}...\n`)
    const { results, wins, avgMinScores } = await runTournament(
      gameCount, playerCount, undefined,
      { aiKinds, onnxModelPath: modelFlag },
    )

    if (log) {
      for (let g = 0; g < results.length; g++) {
        console.log(`\n--- Game ${g + 1} ---`)
        for (const line of results[g].log) console.log(line)
      }
      console.log()
    }

    console.log('Tournament Results:')
    for (let i = 0; i < playerCount; i++) {
      const kindLabel = aiKinds ? ` [${aiKinds[i]}]` : ''
      console.log(
        `  Player ${i + 1} (${results[0].scores[i].dynasty})${kindLabel}: wins=${wins[i]}/${gameCount} (${((wins[i] / gameCount) * 100).toFixed(1)}%) | avg min score=${avgMinScores[i].toFixed(1)}`,
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
}

main().catch(e => {
  console.error(e)
  process.exit(1)
})
