from ultimate_board import UltimateBoard

def terminal_value(game) -> float:
    """
    Terminal value from the perspective of the CURRENT player to move.
    +1 win, 0 draw, -1 loss
    """
    if not game.is_terminal():
        raise ValueError("terminal_value called on non-terminal state")
    return float(game.value_for(game.current_player))
