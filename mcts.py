import math
import random
from small_board import SmallBoard


class MCTSNode:
    def __init__(self, parent=None, move=None, untried_moves=None):
        self.parent = parent
        self.move = move                  
        self.children = {}
        self.Q = 0
        self.N = 0
        self.untried_moves = untried_moves

    def is_fully_expanded(self):
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def best_child(self, c_param=1.414):
        choices_weights = []

        for child in self.children.values():
            exploitation = child.Q / child.N
            exploration = c_param * math.sqrt(math.log(self.N) / child.N)
            uct = exploitation + exploration
            choices_weights.append((uct, child))

        return max(choices_weights, key=lambda item: item[0])[1]


class MCTSPlayer:
    def __init__(self):
        pass

    def choose_move(self, game, iterations):
        root = MCTSNode(
            parent=None,
            move=None,
            untried_moves=game.legal_actions()
        )

        for _ in range(iterations):
            node = root
            scratch_game = game.clone()

            # Selection
            while (
                not node.untried_moves
                and node.children
                and scratch_game.game_status == SmallBoard.ONGOING
            ):
                node = node.best_child()
                scratch_game.apply_action(node.move)

            # Expansion
            if (
                node.untried_moves
                and scratch_game.game_status == SmallBoard.ONGOING
            ):
                move = node.untried_moves.pop()
                scratch_game.apply_action(move)
                child_node = MCTSNode(
                    parent=node,
                    move=move,
                    untried_moves=scratch_game.legal_actions()
                )
                node.children[move] = child_node
                node = child_node

            # Simulation (random rollout)
            while scratch_game.game_status == SmallBoard.ONGOING:
                possible_moves = scratch_game.legal_actions()
                move = random.choice(possible_moves)
                scratch_game.apply_action(move)

            # Backpropagation (IDENTICAL logic)
            result = scratch_game.game_status

            path = []
            temp = node
            while temp is not None:
                path.append(temp)
                temp = temp.parent

            current_player = game.current_player

            for n in reversed(path):
                n.N += 1
                if n.parent is not None:
                    if result == SmallBoard.DRAW:
                        n.Q += 0.5
                    elif (result == SmallBoard.X_WIN and current_player == SmallBoard.X) or \
                        (result == SmallBoard.O_WIN and current_player == SmallBoard.O):
                        n.Q += 1

                    current_player = SmallBoard.O if current_player == SmallBoard.X else SmallBoard.X


        best_move = max(root.children.items(), key=lambda item: item[1].N)[0]
        return best_move
