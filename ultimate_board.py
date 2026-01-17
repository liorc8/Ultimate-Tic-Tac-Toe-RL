from small_board import SmallBoard

class UltimateBoard:
    
    def __init__(self):
        """Initialize an empty 3x3 ultimate board composed of small boards."""
        self.boards = [[SmallBoard() for _ in range(3)] for _ in range(3)]
        self.current_player = SmallBoard.X
        self.game_status = SmallBoard.ONGOING
        self.active_board = None

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




