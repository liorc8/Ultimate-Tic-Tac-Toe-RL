"""
Comprehensive MCTS test runner for Ultimate Tic-Tac-Toe.

What it checks:
1) Smoke tests: MCTS returns a legal action at the start and mid-game.
2) Game completion: MCTS vs Random and MCTS vs MCTS always terminate (no infinite loops).
3) Safety: No illegal actions are played.
4) Clone sanity: clone() doesn't mutate the original when you play on the clone.
5) Summary stats: win/draw rates, average moves, optional timing.

How to run:
    python test_mcts.py

Requirements:
- ultimate_board.py (UltimateBoard)
- small_board.py (SmallBoard)
- mcts.py (MCTSPlayer)  -> should expose MCTSPlayer with choose_move(game, iterations)
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

from src.ultimate_board import UltimateBoard
from src.small_board import SmallBoard
from src.mcts import MCTSPlayer


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MCTS_ITERS = 200
GAMES_MCTS_VS_RANDOM = 50
GAMES_MCTS_VS_MCTS = 20

# Ultimate Tic-Tac-Toe has at most 81 moves (9 boards * 9 cells).
# We keep a little buffer just in case your implementation allows extra moves (it shouldn't).
MAX_PLIES_SAFETY = 100

RANDOM_SEED = 42


# -----------------------------
# Helpers / Players
# -----------------------------
def other_player(p: str) -> str:
    return SmallBoard.O if p == SmallBoard.X else SmallBoard.X


def choose_random_action(game: UltimateBoard) -> int:
    actions = game.legal_actions()
    if not actions:
        raise RuntimeError("No legal actions but game not terminal?")
    return random.choice(actions)


@dataclass
class GameResult:
    status: int
    plies: int
    illegal: bool
    error: Optional[str]
    duration_sec: float


def play_game(
    player_x: Callable[[UltimateBoard], int],
    player_o: Callable[[UltimateBoard], int],
    *,
    max_plies: int = MAX_PLIES_SAFETY,
) -> GameResult:
    game = UltimateBoard()
    start_t = time.perf_counter()

    illegal = False
    error: Optional[str] = None
    plies = 0

    try:
        while game.game_status == SmallBoard.ONGOING:
            if plies >= max_plies:
                raise RuntimeError(f"Exceeded max plies safety limit ({max_plies}). Possible infinite loop.")

            legal = set(game.legal_actions())
            if not legal:
                raise RuntimeError("No legal actions while game is ONGOING.")

            if game.current_player == SmallBoard.X:
                action = player_x(game)
            else:
                action = player_o(game)

            if action not in legal:
                illegal = True
                raise RuntimeError(f"Illegal action played: {action}. Legal count={len(legal)}")

            game.apply_action(action)
            plies += 1

    except Exception as e:
        error = str(e)

    dur = time.perf_counter() - start_t
    return GameResult(status=game.game_status, plies=plies, illegal=illegal, error=error, duration_sec=dur)


# -----------------------------
# Individual Tests
# -----------------------------
def test_clone_independence() -> None:
    g = UltimateBoard()
    g2 = g.clone()

    # Make one move on clone, original should remain unchanged
    a = random.choice(g2.legal_actions())
    g2.apply_action(a)

    # Original should still have 81 legal actions at start
    # (unless your rules change it, but at start it's 81)
    assert g.game_status == SmallBoard.ONGOING
    assert len(g.legal_actions()) == 81, "clone() seems to have mutated the original game."

    print("✅ clone() independence OK")


def test_mcts_returns_legal_action(iterations: int) -> None:
    mcts = MCTSPlayer()
    g = UltimateBoard()

    a = mcts.choose_move(g, iterations)
    assert a in set(g.legal_actions()), "MCTS returned an illegal action at game start."
    print("✅ MCTS returns a legal action at start")

    # Apply one move and test again
    g.apply_action(a)
    a2 = mcts.choose_move(g, iterations)
    assert a2 in set(g.legal_actions()), "MCTS returned an illegal action mid-game."
    print("✅ MCTS returns a legal action mid-game")


def run_matchups(mcts_iters: int) -> None:
    mcts = MCTSPlayer()

    def mcts_policy(game: UltimateBoard) -> int:
        return mcts.choose_move(game, mcts_iters)

    def random_policy(game: UltimateBoard) -> int:
        return choose_random_action(game)

    # ---- MCTS vs Random (MCTS as X) ----
    print(f"\n--- Running MCTS({mcts_iters}) as X vs Random for {GAMES_MCTS_VS_RANDOM} games ---")
    stats = {"X_WIN": 0, "O_WIN": 0, "DRAW": 0, "ERROR": 0}
    plies_list = []
    time_list = []

    for i in range(GAMES_MCTS_VS_RANDOM):
        r = play_game(mcts_policy, random_policy)
        plies_list.append(r.plies)
        time_list.append(r.duration_sec)

        if r.error is not None:
            stats["ERROR"] += 1
            print(f"❌ Game {i+1}: ERROR: {r.error}")
            continue

        if r.status == SmallBoard.X_WIN:
            stats["X_WIN"] += 1
        elif r.status == SmallBoard.O_WIN:
            stats["O_WIN"] += 1
        else:
            stats["DRAW"] += 1

    print("Results:", stats)
    if plies_list:
        print(f"Avg plies: {sum(plies_list)/len(plies_list):.2f}, Max plies: {max(plies_list)}")
    if time_list:
        print(f"Avg game time: {sum(time_list)/len(time_list):.3f}s")

    # ---- Random vs MCTS (MCTS as O) ----
    print(f"\n--- Running Random as X vs MCTS({mcts_iters}) as O for {GAMES_MCTS_VS_RANDOM} games ---")
    stats = {"X_WIN": 0, "O_WIN": 0, "DRAW": 0, "ERROR": 0}
    plies_list = []
    time_list = []

    for i in range(GAMES_MCTS_VS_RANDOM):
        r = play_game(random_policy, mcts_policy)
        plies_list.append(r.plies)
        time_list.append(r.duration_sec)

        if r.error is not None:
            stats["ERROR"] += 1
            print(f"❌ Game {i+1}: ERROR: {r.error}")
            continue

        if r.status == SmallBoard.X_WIN:
            stats["X_WIN"] += 1
        elif r.status == SmallBoard.O_WIN:
            stats["O_WIN"] += 1
        else:
            stats["DRAW"] += 1

    print("Results:", stats)
    if plies_list:
        print(f"Avg plies: {sum(plies_list)/len(plies_list):.2f}, Max plies: {max(plies_list)}")
    if time_list:
        print(f"Avg game time: {sum(time_list)/len(time_list):.3f}s")

    # ---- MCTS vs MCTS ----
    print(f"\n--- Running MCTS({mcts_iters}) vs MCTS({mcts_iters}) for {GAMES_MCTS_VS_MCTS} games ---")
    stats = {"X_WIN": 0, "O_WIN": 0, "DRAW": 0, "ERROR": 0}
    plies_list = []
    time_list = []

    for i in range(GAMES_MCTS_VS_MCTS):
        r = play_game(mcts_policy, mcts_policy)
        plies_list.append(r.plies)
        time_list.append(r.duration_sec)

        if r.error is not None:
            stats["ERROR"] += 1
            print(f"❌ Game {i+1}: ERROR: {r.error}")
            continue

        if r.status == SmallBoard.X_WIN:
            stats["X_WIN"] += 1
        elif r.status == SmallBoard.O_WIN:
            stats["O_WIN"] += 1
        else:
            stats["DRAW"] += 1

    print("Results:", stats)
    if plies_list:
        print(f"Avg plies: {sum(plies_list)/len(plies_list):.2f}, Max plies: {max(plies_list)}")
    if time_list:
        print(f"Avg game time: {sum(time_list)/len(time_list):.3f}s")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    random.seed(RANDOM_SEED)

    print("=== MCTS Comprehensive Test Suite ===")
    print(f"Seed={RANDOM_SEED}")
    print(f"MCTS iterations per move={DEFAULT_MCTS_ITERS}")

    # 1) Basic correctness checks
    test_clone_independence()
    test_mcts_returns_legal_action(DEFAULT_MCTS_ITERS)

    # 2) Full game stress tests
    run_matchups(DEFAULT_MCTS_ITERS)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
