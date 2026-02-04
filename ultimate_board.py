from small_board import SmallBoard
from codec import encode_action, decode_action


class UltimateBoard:
    
    def __init__(self):
        """Initialize an empty 3x3 ultimate board composed of small boards."""
        self.boards = [[SmallBoard() for _ in range(3)] for _ in range(3)]
        self.current_player = SmallBoard.X
        self.game_status = SmallBoard.ONGOING
        self.active_board = None


    def clone(self):
        """
        Return a deep copy of the ultimate board."""
        cloned = UltimateBoard()

        cloned.boards = [
            [self.boards[r][c].clone() for c in range(3)]
            for r in range(3)
        ]

        cloned.current_player = self.current_player
        cloned.game_status = self.game_status

        cloned.active_board = None if self.active_board is None else (self.active_board[0], self.active_board[1])

        return cloned


    def legal_moves(self):
        """
        Return a list of all legal moves as (br, bc, sr, sc).
        """
        if self.game_status != SmallBoard.ONGOING:
            return []
        moves = []

        if self.active_board is not None:
            br, bc = self.active_board
            small = self.boards[br][bc]

            if small.is_open():
                for (sr, sc) in small.legal_moves():
                    moves.append((br, bc, sr, sc))
                return moves

        for br in range(3):
            for bc in range(3):
                small = self.boards[br][bc]
                if small.is_open():
                    for (sr, sc) in small.legal_moves():
                        moves.append((br, bc, sr, sc))

        return moves


    def is_terminal(self):
        """Return True if the ultimate game is finished (win or draw)."""
        return self.game_status != SmallBoard.ONGOING


    def result(self):
        """Return the current game result/status (X_WIN, O_WIN, DRAW, or ONGOING)."""
        return self.game_status


    def value_for(self, player):
        """
        Return the terminal value from the perspective of `player`:
        +1 if `player` wins, -1 if `player` loses, 0 for draw or non-terminal.
        """
        if self.game_status == SmallBoard.ONGOING:
            return 0

        if self.game_status == SmallBoard.DRAW:
            return 0

        if self.game_status == SmallBoard.X_WIN:
            return 1 if player == SmallBoard.X else -1

        if self.game_status == SmallBoard.O_WIN:
            return 1 if player == SmallBoard.O else -1

        # Safety fallback (shouldn't happen)
        return 0


    def make_move(self, br, bc, sr, sc):
        """
        Apply a move in small board (br, bc) at cell (sr, sc) for current_player.
        """

        if self.game_status != SmallBoard.ONGOING:
            raise ValueError("Ultimate game is already over")

        if not (0 <= br < 3 and 0 <= bc < 3 and 0 <= sr < 3 and 0 <= sc < 3):
            raise ValueError("Move out of bounds")

        if self.active_board is not None and (br, bc) != self.active_board:
            raise ValueError(f"Must play in active board {self.active_board}")

        small = self.boards[br][bc]
        if not small.is_open():
            raise ValueError("Target small board is closed")

        if not small.is_empty(sr, sc):
            raise ValueError("Cell is not empty")

        small.place_mark(sr, sc, self.current_player)

        self.update_status()

        if self.game_status == SmallBoard.ONGOING:
            self.set_next_active_board(sr, sc)
            self.current_player = SmallBoard.O if self.current_player == SmallBoard.X else SmallBoard.X


    def apply_action(self, action):
        """
        Apply an encoded action (0..80) by decoding it and calling make_move.
        """
        br, bc, sr, sc = decode_action(action)
        self.make_move(br, bc, sr, sc)


    def legal_actions(self):
        """
        Return legal moves as encoded actions in [0, 80].
        """
        return [encode_action(br, bc, sr, sc) for (br, bc, sr, sc) in self.legal_moves()]


    def meta_cell(self, br, bc):
        """
        Return the 'owner' of small board (br, bc) for meta-board:
        """
        
        if not (0 <= br < 3 and 0 <= bc < 3):
            raise ValueError("Board indices out of bounds")

        sb = self.boards[br][bc]
        if sb.status == SmallBoard.X_WIN:
            return SmallBoard.X
        if sb.status == SmallBoard.O_WIN:
            return SmallBoard.O
        return SmallBoard.EMPTY


    def update_status(self):
        """
        Update the overall game_status based on small boards' statuses.
        """
        meta = [[self.meta_cell(r, c) for c in range(3)] for r in range(3)]

        for i in range(3):
            # rows
            if meta[i][0] == meta[i][1] == meta[i][2] != SmallBoard.EMPTY:
                self.game_status = SmallBoard.X_WIN if meta[i][0] == SmallBoard.X else SmallBoard.O_WIN
                return

            # columns
            if meta[0][i] == meta[1][i] == meta[2][i] != SmallBoard.EMPTY:
                self.game_status = SmallBoard.X_WIN if meta[0][i] == SmallBoard.X else SmallBoard.O_WIN
                return

        # diagonals
        if meta[0][0] == meta[1][1] == meta[2][2] != SmallBoard.EMPTY:
            self.game_status = SmallBoard.X_WIN if meta[0][0] == SmallBoard.X else SmallBoard.O_WIN
            return

        if meta[0][2] == meta[1][1] == meta[2][0] != SmallBoard.EMPTY:
            self.game_status = SmallBoard.X_WIN if meta[0][2] == SmallBoard.X else SmallBoard.O_WIN
            return

        # draw
        all_closed = all(self.boards[r][c].status != SmallBoard.ONGOING for r in range(3) for c in range(3))
        self.game_status = SmallBoard.DRAW if all_closed else SmallBoard.ONGOING


    def set_next_active_board(self, sr, sc):
        """
        Set the next active small board based on last move's (sr, sc).
        """
        if not (0 <= sr < 3 and 0 <= sc < 3):
            raise ValueError("Target small-board index out of bounds")

        target = self.boards[sr][sc]
        if target.status == SmallBoard.ONGOING:
            self.active_board = (sr, sc)
        else:
            self.active_board = None


    def __str__(self):
        def small_row_to_str(sb, r):
            return " ".join(sb.board[r])

        lines = []
        lines.append(f"Player: {self.current_player} | Active: {self.active_board} | Status: {self.game_status}")
        lines.append("")

        # Print 3 big rows; each big row contains 3 small boards
        for big_r in range(3):
            for small_r in range(3):
                row_parts = []
                for big_c in range(3):
                    sb = self.boards[big_r][big_c]
                    row_parts.append(small_row_to_str(sb, small_r))
                lines.append(" || ".join(row_parts))
            if big_r < 2:
                lines.append("======++=======++======")

        return "\n".join(lines)




