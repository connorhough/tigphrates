# Tigris & Euphrates - Complete Game Rules Reference

## Overview
- Designer: Reiner Knizia (1997)
- Players: 2-4
- Theme: Competing dynasties in ancient Mesopotamia

## Components

### Civilization Tiles (153 total)
| Color | Type | Count | Placement |
|-------|------|-------|-----------|
| Red | Temple | 57 | Land only |
| Blue | Farm | 36 | River only |
| Green | Market | 30 | Land only |
| Black | Settlement | 30 | Land only |

### Leaders (4 per player, 4 dynasties)
Dynasties identified by symbol: Archer, Bull, Pot, Lion.

| Color | Leader | Domain |
|-------|--------|--------|
| Black | King | Settlements (wildcard for tile scoring) |
| Red | Priest | Temples |
| Blue | Farmer | Farms |
| Green | Trader | Markets |

### Other Components
- **4 Unification Tiles** (one per dynasty)
- **8 Catastrophe Tiles** (2 per player)
- **6 Monuments** (one for each 2-color combination: R-Bl, R-G, R-Bk, Bl-G, Bl-Bk, G-Bk)
- **Victory Point Tokens**: 4 colors, plenty of each
- **10 Treasure Tokens** (standard board) / 14 (advanced board)
- **1 Cloth Bag**

## The Board

### Dimensions
**16 columns x 11 rows** grid.

### Space Types
- **Land**: Temples (red), settlements (black), markets (green), and leaders go here.
- **River**: Only farms (blue) can be placed here.

### Standard Board Layout
```
Row  0: .  .  .  .  x  x  x  x  x  .  T  .  x  .  .  .
Row  1: .  T  .  .  x  .  .  .  .  .  .  .  x  .  .  T
Row  2: .  .  .  x  x  T  .  .  .  .  .  .  x  x  .  .
Row  3: x  x  x  x  .  .  .  .  .  .  .  .  .  x  x  x
Row  4: .  .  .  .  .  .  .  .  .  .  .  .  .  T  x  x
Row  5: .  .  .  .  .  .  .  .  .  .  .  .  .  .  x  .
Row  6: x  x  x  x  .  .  .  .  .  T  .  .  x  x  x  .
Row  7: .  T  .  x  x  x  x  x  .  .  .  .  x  .  .  .
Row  8: .  .  .  .  .  .  .  x  x  x  x  x  x  .  T  .
Row  9: .  .  .  .  .  .  T  .  .  .  .  .  .  .  .  .
Row 10: .  .  .  .  .  .  .  .  .  .  T  .  .  .  .  .
```
Key: `.` = land, `x` = river, `T` = starting temple (with treasure)

### 10 Starting Temple Positions (col, row)
(10,0), (1,1), (15,1), (5,2), (13,4), (9,6), (1,7), (14,8), (6,9), (10,10)

## Setup

1. Place red temple tile + treasure on each of the 10 starting positions.
2. Put remaining 143 tiles in the bag.
3. Each player takes: 4 leaders, 1 unification tile, 2 catastrophe tiles, 1 screen.
4. Each player draws 6 tiles from the bag (hidden).
5. Leaders, unification tile, catastrophe tiles are public info.
6. Random starting player, play proceeds clockwise.

## Key Concepts

### Region
Connected tiles with NO leaders.

### Kingdom
Connected tiles AND leaders (at least one leader present).

### Adjacency
Sharing an edge (up/down/left/right). No diagonals. River tiles with farms connect to adjacent land tiles -- river/land only restricts what can be *placed*, not connectivity.

### Face-down tiles (under monuments)
Still count for connectivity. Do NOT count as supporters in conflicts or as temples for leader adjacency.

## Turn Structure

**2 actions per turn** (may repeat same action). Four possible actions:

1. **Place/Move/Withdraw a Leader**
2. **Place a Civilization Tile**
3. **Place a Catastrophe Tile**
4. **Swap Tiles** (discard up to 6, draw replacements from bag)

After actions: draw tiles to refill hand to 6, then monument VP scoring, then treasure collection, then game end check.

## Action: Place/Move/Withdraw Leader

### Rules
- Must be placed on an **empty land space**.
- Must be **adjacent to at least one face-up red temple tile**.
- **Cannot unite two separate kingdoms**.
- Can come from off-board or be repositioned from elsewhere on board.
- Withdrawing a leader (returning to supply) counts as an action.

### Forced Withdrawal
If a leader ever loses adjacency to all face-up red temples, it is immediately withdrawn.

### Internal Conflict (Revolt)
Triggered when placing a leader into a kingdom that already has a same-colored leader from another player.

## Internal Conflict (Revolt)

### Resolution
1. **Base strength**: Count face-up red temple tiles adjacent to each involved leader.
2. **Support from hand**: Attacker first, then defender, may commit red temple tiles from hand.
3. **Compare**: Higher total wins. **Ties go to defender.**
4. **Aftermath**:
   - Loser's leader returned to owner.
   - Winner gains **1 red VP**.
   - All committed hand tiles removed from game (to box).
   - Board tiles are NOT removed.

### Key: Only red tiles matter in revolts, regardless of leader colors involved.

## Action: Place a Civilization Tile

### Rules
- Blue tiles: river spaces only. Red/Green/Black: land spaces only.
- Cannot place on occupied spaces.
- **Cannot unite more than two kingdoms** in one placement.

### Scoring
- If kingdom has a leader matching the tile's color: that leader's owner gets **1 VP of that color**.
- If no matching leader but a **King (black)** exists: King's owner gets **1 black VP**.
- If neither: no points.
- **No VP scored when a tile unites two kingdoms** (war overrides scoring).

### External Conflict (War)
Triggered when tile placement unites two kingdoms and the merged kingdom has two leaders of the same color.

## External Conflict (War)

### Setup
- The placed tile is covered by the active player's **unification tile** (doesn't count as supporter).
- If multiple color conflicts exist, **active player chooses resolution order**.

### Attacker/Defender
- If active player has one of the conflicting leaders, they are attacker.
- Otherwise, next player clockwise with an involved leader is attacker.

### Resolution
1. **Base strength**: Count tiles of the **conflict color** on each side of the kingdom (connected to each leader, separated by the unification tile).
2. **Support from hand**: Attacker then defender commit tiles of the **conflict color**.
3. **Compare**: Higher wins. **Ties go to defender.**
4. **Aftermath**:
   - Loser's leader returned to owner.
   - **All tiles of the conflict color on the loser's side** are removed from the game.
   - Treasures on removed tiles return to general supply.
   - Winner gains **1 VP per removed tile + 1 for removed leader** (in conflict color).
   - Committed hand tiles removed from game.
   - Re-evaluate board: kingdom may split, leaders may lose temple adjacency.

### After all wars resolve
Remove unification tile, reveal original tile underneath. Check for monument building.

### Red War Special Rule
In wars involving priests, red temples with treasures on them AND red temples adjacent to another leader cannot be removed as casualties.

## Monuments

### Building
After tile placement (and conflict resolution), if a **2x2 square of same-colored face-up tiles** exists:
1. Flip the 4 tiles face-down.
2. Place an available monument with one color matching the flipped tiles.
3. Only the active player can build; if declined, opportunity is lost for that arrangement.
4. Monuments are **permanent and indestructible**.

### Effects
- Face-down tiles still connect but don't count as supporters or temples.
- If a leader loses temple adjacency from the flip, it's immediately withdrawn.

### Monument VP Scoring
At end of active player's turn: for each of their leaders in a kingdom with a color-matching monument, gain **1 VP of that color per matching monument**. King only scores from black component (no wildcard).

## Treasures

### Collection
At end of turn: if a kingdom has **2+ treasures** and a **green Trader**, the Trader's owner takes all but one. Treasures on special (starting) spaces must be taken first.

### Value
Each treasure is a **wildcard VP** -- assigned to any color at game end.

## Catastrophe Tiles

- 2 per player for the entire game.
- Place on any empty space or on a face-up tile (destroying it).
- **Cannot** place on: leaders, treasures, monuments, or existing catastrophes.
- Permanently occupies the space. Breaks connectivity. Can split kingdoms.

## Swap Tiles

- Discard 1-6 tiles from hand to the box (out of game).
- Draw equal number from bag.
- Counts as 1 action.

## Game End

Game ends immediately when either:
1. Only **1 or 2 treasures** remain on the board (after treasure collection).
2. Not enough tiles in bag to refill a player's hand.

### Final Scoring
1. Assign treasure tokens (wildcards) to any colors.
2. Each player's score = their **lowest color**.
3. Highest minimum wins.
4. Tiebreak: compare second-lowest, then third, then highest. Full tie = draw.

## Complete Turn Sequence

1. **Action 1** (resolve conflicts + monument check if tile placed)
2. **Action 2** (resolve conflicts + monument check if tile placed)
3. **Draw** to 6 tiles
4. **Monument scoring** for active player's leaders
5. **Treasure collection** in all kingdoms
6. **Game end check**
7. Next player
