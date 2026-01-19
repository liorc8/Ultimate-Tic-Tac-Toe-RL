from small_board import SmallBoard
from ultimate_board import UltimateBoard


def play_small_board_cli():
    """CLI runner for SmallBoard (human vs human)."""
    game = SmallBoard()

    while game.status == game.ONGOING:
        print("\n" + str(game))
        print(f"\nTurn: {game.player}")
        raw = input("Enter move as 'row col' (0-2 0-2): ").strip()

        try:
            row_s, col_s = raw.split()
            row, col = int(row_s), int(col_s)
            game.make_move(row, col)
        except ValueError as e:
            print(f"❌ {e}")
            continue

    print("\n" + str(game))
    if game.status == game.X_WIN:
        print("\n🏆 X wins!")
    elif game.status == game.O_WIN:
        print("\n🏆 O wins!")
    else:
        print("\n🤝 Draw!")


def play_ultimate_board_cli():
    """CLI runner for UltimateBoard (human vs human)."""
    game = UltimateBoard()

    while game.game_status == SmallBoard.ONGOING:
        print("\n" + str(game))
        print(f"\nTurn: {game.current_player}")

        if game.active_board is None:
            print("You can play in ANY open small board.")
        else:
            print(f"You MUST play in small board: {game.active_board} (br bc)")

        raw = input("Enter move as 'br bc sr sc' (0-2 0-2 0-2 0-2): ").strip()

        try:
            br_s, bc_s, sr_s, sc_s = raw.split()
            br, bc, sr, sc = int(br_s), int(bc_s), int(sr_s), int(sc_s)
            game.make_move(br, bc, sr, sc)
        except ValueError as e:
            print(f"❌ {e}")
            continue

    print("\n" + str(game))
    if game.game_status == SmallBoard.X_WIN:
        print("\n🏆 X wins the Ultimate game!")
    elif game.game_status == SmallBoard.O_WIN:
        print("\n🏆 O wins the Ultimate game!")
    else:
        print("\n🤝 Ultimate game ended in a draw!")


if __name__ == "__main__":
    choice = input("Choose game: (1) SmallBoard (2) UltimateBoard : ").strip()
    if choice == "1":
        play_small_board_cli()
    else:
        play_ultimate_board_cli()
