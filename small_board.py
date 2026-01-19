class SmallBoard:
    # Player constants
    X = 'X'
    O = 'O'
    EMPTY = ' '

    # Game status constants
    X_WIN = 1
    O_WIN = -1
    DRAW = 0
    ONGOING = -17


    def __init__(self):
        """Initialize an empty 3x3 board and set the starting player."""

        self.board = [[self.EMPTY for _ in range(3)] for _ in range(3)]
        self.player = self.X
        self.status = self.ONGOING

    def legal_moves(self):
        """Return a list of legal moves as (row, column) tuples."""
        if self.status != self.ONGOING:
            return []

        moves = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == self.EMPTY:
                    moves.append((r, c))
        return moves

    def is_empty(self, row, col):
        """Return True if (row, col) is inside bounds and currently EMPTY."""
        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError("Move out of bounds")
        return self.board[row][col] == self.EMPTY

    def is_open(self):
        """Return True if this small board can still accept moves."""
        return self.status == self.ONGOING

    def make_move(self, row, col):
        """Place the current player's mark on the board at (row, col)
        
        Args:
            row (int): The row index (0-2).
            col (int): The column index (0-2).
        """
        self.place_mark(row, col, self.player)
        if self.status == self.ONGOING:
            self.player = self.other_player(self.player)

    def place_mark(self, row, col, player):
        """
        Place a specific player's mark (X/O) at (row, col).
        This does NOT change turns. Intended for UltimateBoard (global turn).

        Args:
            row (int): The row index (0-2).
            col (int): The column index (0-2).
            player (str): The player making the move ('X' or 'O').
        """
        if self.status != self.ONGOING:
            raise ValueError("Board is already over")

        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError("Move out of bounds")

        if self.board[row][col] != self.EMPTY:
            raise ValueError("Cell is not empty")

        if player not in (self.X, self.O):
            raise ValueError("Invalid player")

        self.board[row][col] = player
        self.winning_move()

    def unmake_move(self, row, col):
        """Remove the mark from the board at (row, col) and revert the player turn.
        
        Args:
            row (int): The row index (0-2).
            col (int): The column index (0-2).  
        """
        self.board[row][col] = self.EMPTY
        self.player = self.other_player(self.player)
        self.status = self.ONGOING


    def other_player(self, player):
        """Return the opposite player."""
        return self.O if player == self.X else self.X

    def winning_move(self):
        """Check the board for a win or draw and update the game status."""
        lines = []

        for i in range(3):
            # rows
            lines.append(self.board[i])
            # columns
            lines.append([self.board[r][i] for r in range(3)])

        # diagonals
        lines.append([self.board[i][i] for i in range(3)])
        lines.append([self.board[i][2 - i] for i in range(3)])

        for line in lines:
            if all(cell == self.X for cell in line):
                self.status = self.X_WIN
                return

            if all(cell == self.O for cell in line):
                self.status = self.O_WIN
                return

        if all(cell != self.EMPTY for row in self.board for cell in row):
            self.status = self.DRAW


    def clone(self):
        clone = SmallBoard()
        clone.board = [row[:] for row in self.board]
        clone.player = self.player
        clone.status = self.status
        return clone

    
    def __str__(self):
        rows = []
        for r in range(3):
            rows.append('|'.join(self.board[r]))
        return '\n-----\n'.join(rows)

