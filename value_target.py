from ultimate_board import UltimateBoard
from small_board import SmallBoard


def terminal_value(game: UltimateBoard) -> float:
    """
    Return the terminal value from the perspective of the current player.

    +1.0 -> current player wins
     0.0 -> draw
    -1.0 -> current player loses

    Should be called ONLY if game.is_terminal() is True.
    """
    if not game.is_terminal():
        raise ValueError("terminal_value called on non-terminal state")

    result = game.result()

    if result == SmallBoard.DRAW:
        return 0.0

    if result == game.current_player:
        return 1.0
    else:
        return -1.0