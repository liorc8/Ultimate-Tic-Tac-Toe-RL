import time
import random
from dataclasses import dataclass
from typing import Callable, Optional

from ultimate_board import UltimateBoard
from small_board import SmallBoard
from game_network import GameNetwork
from puct_player import PUCTPlayer


# -----------------------------
# Config
# -----------------------------
SEED = 42

PUCT_SIMS = 50
GAMES_PUCT_VS_RANDOM = 30
GAMES_PUCT_VS_PUCT = 10

MAX_PLIES_SAFETY = 100  # Ultimate TTT should finish within <=81


# -----------------------------
# Helpers
# -----------------------------
def random_action(game: UltimateBoard) -> int:
    legal = game.legal_actions()
    if not legal:
        raise RuntimeError("No legal actions while game is ongoing")
    return random.choice(legal)


@dataclass
class GameResult:
    status: int
    plies: int
    error: Optional[str]
    duration_sec: float


def play_game(player_x: Callable[[UltimateBoard], int],
              player_o: Callable[[UltimateBoard], int]) -> GameResult:
    g = UltimateBoard()
    start = time.perf_counter()
    plies = 0
    err = None

    try:
        while g.game_status == SmallBoard.ONGOING:
            if plies >= MAX_PLIES_SAFETY:
                raise RuntimeError(f"Exceeded max plies safety limit ({MAX_PLIES_SAFETY})")

            legal = set(g.legal_actions())
            if not legal:
                raise RuntimeError("No legal actions but status is ONGOING")

            a = player_x(g) if g.current_player == SmallBoard.X else player_o(g)

            if a not in legal:
                raise RuntimeError(f"Illegal action {a}. Legal count={len(legal)}")

            g.apply_action(a)
            plies += 1

    except Exception as e:
        err = str(e)

    dur = time.perf_counter() - start
    return GameResult(status=g.game_status, plies=plies, error=err, duration_sec=dur)


def main():
    random.seed(SEED)

    print("=== PUCT Comprehensive Test Suite ===")
    print(f"Seed={SEED}")
    print(f"PUCT simulations per move={PUCT_SIMS}")

    # ----- Build Dummy Network + PUCT -----
    net = GameNetwork.load("game_network_pretrained.pt", map_location="cpu")
    puct = PUCTPlayer(net, c_puct=1.0)


    def puct_policy(game: UltimateBoard) -> int:
        return puct.choose_move(game, PUCT_SIMS)

    def rnd_policy(game: UltimateBoard) -> int:
        return random_action(game)

    # ----- Smoke test: choose_move returns legal at start -----
    g0 = UltimateBoard()
    a0 = puct.choose_move(g0, PUCT_SIMS)
    assert a0 in set(g0.legal_actions()), "PUCT returned illegal action at start"
    print("✅ PUCT returns legal action at start")

    # ----- Smoke test: midgame -----
    g0.apply_action(a0)
    a1 = puct.choose_move(g0, PUCT_SIMS)
    assert a1 in set(g0.legal_actions()), "PUCT returned illegal action midgame"
    print("✅ PUCT returns legal action midgame")

    # ----- Smoke test: choose_move_with_pi -----
    g_test = UltimateBoard()
    a, pi = puct.choose_move_with_pi(g_test, PUCT_SIMS)

    assert pi.shape == (81,), "pi must have shape (81,)"
    assert abs(float(pi.sum()) - 1.0) < 1e-5, f"pi must sum to 1, got {pi.sum()}"
    assert a in set(g_test.legal_actions()), "best action from pi is illegal"
    assert pi[a] == pi.max(), "best action should match argmax(pi)"

    print("✅ PUCT choose_move_with_pi returns valid (action, pi)")


    # ----- PUCT vs Random -----
    print(f"\n--- Running PUCT({PUCT_SIMS}) as X vs Random for {GAMES_PUCT_VS_RANDOM} games ---")
    stats = {"X_WIN": 0, "O_WIN": 0, "DRAW": 0, "ERROR": 0}
    times = []
    plies_list = []

    for i in range(GAMES_PUCT_VS_RANDOM):
        r = play_game(puct_policy, rnd_policy)
        times.append(r.duration_sec)
        plies_list.append(r.plies)

        if r.error:
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
    print(f"Avg plies: {sum(plies_list)/len(plies_list):.2f}, Max plies: {max(plies_list)}")
    print(f"Avg game time: {sum(times)/len(times):.3f}s")

    # ----- Random vs PUCT -----
    print(f"\n--- Running Random as X vs PUCT({PUCT_SIMS}) as O for {GAMES_PUCT_VS_RANDOM} games ---")
    stats = {"X_WIN": 0, "O_WIN": 0, "DRAW": 0, "ERROR": 0}
    times = []
    plies_list = []

    for i in range(GAMES_PUCT_VS_RANDOM):
        r = play_game(rnd_policy, puct_policy)
        times.append(r.duration_sec)
        plies_list.append(r.plies)

        if r.error:
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
    print(f"Avg plies: {sum(plies_list)/len(plies_list):.2f}, Max plies: {max(plies_list)}")
    print(f"Avg game time: {sum(times)/len(times):.3f}s")

    # ----- PUCT vs PUCT -----
    print(f"\n--- Running PUCT({PUCT_SIMS}) vs PUCT({PUCT_SIMS}) for {GAMES_PUCT_VS_PUCT} games ---")
    stats = {"X_WIN": 0, "O_WIN": 0, "DRAW": 0, "ERROR": 0}
    times = []
    plies_list = []

    for i in range(GAMES_PUCT_VS_PUCT):
        r = play_game(puct_policy, puct_policy)
        times.append(r.duration_sec)
        plies_list.append(r.plies)

        if r.error:
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
    print(f"Avg plies: {sum(plies_list)/len(plies_list):.2f}, Max plies: {max(plies_list)}")
    print(f"Avg game time: {sum(times)/len(times):.3f}s")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
