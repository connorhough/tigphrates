"""
Debug script: play one full game via bridge (heuristic vs heuristic) and log every move.
Confirms engine is functioning correctly from Python side.

Usage:
    python python/debug_game.py [--games N]
"""
import sys
import json
import argparse

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tigphrates_env import BridgeProcess

COLORS = ["red", "blue", "green", "black"]


def fmt_scores(scores):
    parts = []
    for i, s in enumerate(scores):
        mn = s.get("min", 0)
        tr = s.get("treasures", 0)
        parts.append(f"P{i}={mn+tr}({s.get('red',0)}r/{s.get('blue',0)}b/{s.get('green',0)}g/{s.get('black',0)}k+{tr}t)")
    return "  ".join(parts)


def fmt_action(action):
    t = action.get("type", "?")
    if t in ("placeTile", "placeCatastrophe"):
        pos = action.get("position", {})
        color = action.get("color", "")
        return f"{t}({color} @{pos.get('row')},{pos.get('col')})"
    if t == "placeLeader":
        pos = action.get("position", {})
        return f"placeLeader({action.get('color')} @{pos.get('row')},{pos.get('col')})"
    if t == "withdrawLeader":
        return f"withdrawLeader({action.get('color')})"
    if t == "swapTiles":
        return f"swapTiles(mask={action.get('swapMask')})"
    if t == "commitSupport":
        return f"commitSupport(tiles={action.get('supportTiles')})"
    if t == "chooseWarOrder":
        return f"chooseWarOrder({action.get('color')})"
    if t == "buildMonument":
        return f"buildMonument({action.get('color')} @{action.get('position',{})})"
    if t == "declineMonument":
        return "declineMonument"
    if t == "pass":
        return "pass"
    return json.dumps(action)


def play_game(bridge, game_num=1, verbose=True):
    result = bridge.call("create", {"playerCount": 2})
    gid = result["gameId"]

    step = 0
    final_scores = []
    while True:
        va = bridge.call("valid_actions", {"gameId": gid})
        phase = va.get("turnPhase") or va.get("phase", "?")
        active = va.get("activePlayer", "?")

        if phase == "gameOver":
            state_result = bridge.call("get_state", {"gameId": gid})
            raw_state = state_result.get("state", {})
            players = raw_state.get("players", [])
            if players:
                final_scores = [
                    {
                        "red": p["score"]["red"],
                        "blue": p["score"]["blue"],
                        "green": p["score"]["green"],
                        "black": p["score"]["black"],
                        "treasures": p.get("treasures", 0),
                        "min": min(p["score"]["red"], p["score"]["blue"], p["score"]["green"], p["score"]["black"]),
                    }
                    for p in players
                ]
            break

        ai = bridge.call("ai_action", {"gameId": gid})
        action = ai["action"]

        step_result = bridge.call("step_action", {"gameId": gid, "action": action, "playerIndex": active})

        if verbose:
            action_str = fmt_action(action)
            reward = step_result.get("reward", 0)
            reward_str = f" reward={reward:+.1f}" if reward != 0 else ""
            print(f"  step {step:4d} | P{active} [{phase:18s}] {action_str}{reward_str}")

        step += 1
        if step_result.get("done"):
            state_result = bridge.call("get_state", {"gameId": gid})
            raw_state = state_result.get("state", {})
            players = raw_state.get("players", [])
            if players:
                final_scores = [
                    {
                        "red": p["score"]["red"],
                        "blue": p["score"]["blue"],
                        "green": p["score"]["green"],
                        "black": p["score"]["black"],
                        "treasures": p.get("treasures", 0),
                        "min": min(p["score"]["red"], p["score"]["blue"], p["score"]["green"], p["score"]["black"]),
                    }
                    for p in players
                ]
            break
        if step > 5000:
            print("  !! safety limit hit")
            break

    print(f"\nGame {game_num} complete: {step} steps")
    if final_scores:
        print(f"  Scores: {fmt_scores(final_scores)}")
        mins = [s.get("min", 0) + s.get("treasures", 0) for s in final_scores]
        winner = mins.index(max(mins))
        print(f"  Winner: P{winner} (min={mins[winner]})")
    else:
        print("  (no final scores — game may not have ended cleanly)")
    print()

    return {"steps": step, "scores": final_scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Starting bridge...")
    bridge = BridgeProcess()
    print(f"Bridge ready. Playing {args.games} game(s)...\n")

    for g in range(1, args.games + 1):
        print(f"=== Game {g} ===")
        result = play_game(bridge, game_num=g, verbose=not args.quiet)

    bridge.close()
    print("Done.")


if __name__ == "__main__":
    main()
