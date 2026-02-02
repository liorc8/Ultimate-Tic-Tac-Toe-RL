import math


class PUCTNode:
    def __init__(self, parent=None, move=None, prior=0.0):
        """
        A node in the PUCT search tree.

        Args:
            parent: Parent PUCTNode (None for root)
            move: Action (0..80) that led to this node
            prior: Prior probability P(a|s) from the neural network
        """
        self.parent = parent
        self.move = move          # action index (0..80)
        self.P = prior            # prior probability from policy network

        self.children = {}        # action -> PUCTNode

        self.N = 0                # visit count
        self.W = 0.0              # total value (sum of values)
        self.Q = 0.0              # mean value (W / N)

    def update(self, value: float):
        """
        Update this node with a new evaluation value.

        Args:
            value: value in [-1, 1] from the perspective of the current player
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def puct_score(self, c_puct: float) -> float:
        """
        Compute the PUCT score used for child selection.

        PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)

        Args:
            c_puct: exploration constant

        Returns:
            The PUCT score for this node.
        """
        if self.parent is None:
            return self.Q

        exploration = (
            c_puct
            * self.P
            * math.sqrt(self.parent.N)
            / (1 + self.N)
        )

        return self.Q + exploration
