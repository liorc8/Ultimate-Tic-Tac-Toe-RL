from ultimate_board import UltimateBoard
from small_board import SmallBoard

def terminal_value(game: UltimateBoard) -> float:
    """
    Return the terminal value from the perspective of the player who would be
    next to move.
    Returns:
        +1.0 for win, 0.0 draw, -1.0 loss (from the perspective of the next player).
    """
    if not game.is_terminal():
        raise ValueError("terminal_value called on non-terminal state")

    next_player = SmallBoard.O if game.current_player == SmallBoard.X else SmallBoard.X
    return float(game.value_for(next_player))
