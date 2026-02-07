#!/usr/bin/env python3
"""
Human vs Model CLI for Ultimate Tic-Tac-Toe using the existing ASCII drawing (__str__) from UltimateBoard.

Usage:
  python play_human_vs_model.py --weights game_network_pretrained.pt --sims 200 --human X

Notes:
- Human enters moves as: br bc sr sc  (all 0-2)
- The model plays using PUCT guided by the loaded GameNetwork weights.
"""
import argparse
import sys

import torch

from small_board import SmallBoard
from ultimate_board import UltimateBoard
from codec import decode_action
from game_network import GameNetwork
from puct_player import PUCTPlayer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play Ultimate Tic-Tac-Toe: Human vs Model (PUCT+NN).")
    p.add_argument("--weights", type=str, default="game_network_pretrained.pt",
                   help="Path to trained GameNetwork weights (.pt).")
    p.add_argument("--sims", type=int, default=200,
                   help="PUCT simulations per AI move.")
    p.add_argument("--c_puct", type=float, default=1.0,
                   help="PUCT exploration constant.")
    p.add_argument("--human", type=str, default="X", choices=["X", "O"],
                   help="Which side the human plays.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device to run the network on.")
    return p.parse_args()


def pick_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    if arg == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_network(weights_path: str, device: str) -> GameNetwork:
    net = GameNetwork()
    sd = torch.load(weights_path, map_location=device)
    try:
        net.load_state_dict(sd, strict=True)
    except Exception as e:
        # Be permissive if someone saved a full checkpoint dict
        if isinstance(sd, dict) and "state_dict" in sd:
            net.load_state_dict(sd["state_dict"], strict=False)
        else:
            # As a last resort, try non-strict
            net.load_state_dict(sd, strict=False)
    net.to(device)
    net.eval()
    return net


def print_result(game: UltimateBoard) -> None:
    print("\n" + str(game))
    if game.game_status == SmallBoard.X_WIN:
        print("\n🏆 X wins the Ultimate game!")
    elif game.game_status == SmallBoard.O_WIN:
        print("\n🏆 O wins the Ultimate game!")
    else:
        print("\n🤝 Ultimate game ended in a draw!")


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)

    print("=== Human vs Model (PUCT + NN) ===")
    print(f"Weights:  {args.weights}")
    print(f"Device:   {device}")
    print(f"Human:    {args.human}")
    print(f"PUCT sims:{args.sims}")
    print(f"c_puct:   {args.c_puct}")

    try:
        net = load_network(args.weights, device)
    except FileNotFoundError:
        print(f"❌ Weights file not found: {args.weights}", file=sys.stderr)
        return 2

    ai = PUCTPlayer(net, c_puct=args.c_puct)
    human = args.human

    game = UltimateBoard()

    while game.game_status == SmallBoard.ONGOING:
        print("\n" + str(game))
        print(f"\nTurn: {game.current_player}")

        if game.active_board is None:
            print("Active board: ANY open small board.")
        else:
            print(f"Active board: MUST play in {game.active_board} (br bc)")

        if game.current_player == human:
            raw = input("Your move 'br bc sr sc' (0-2 0-2 0-2 0-2): ").strip()
            try:
                br_s, bc_s, sr_s, sc_s = raw.split()
                br, bc, sr, sc = int(br_s), int(bc_s), int(sr_s), int(sc_s)
                game.make_move(br, bc, sr, sc)
            except Exception as e:
                print(f"❌ {e}")
                continue
        else:
            action = ai.choose_move(game, simulations=args.sims)
            br, bc, sr, sc = decode_action(action)
            print(f"🤖 Model plays: {br} {bc} {sr} {sc}  (action={action})")
            try:
                game.apply_action(action)
            except Exception as e:
                print(f"❌ Model produced illegal move?! {e}", file=sys.stderr)
                return 3

    print_result(game)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
