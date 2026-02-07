import numpy as np
import random
from puct_node import PUCTNode
from encoder import encode_state
from action_mask import legal_action_mask
from small_board import SmallBoard
from value_target import terminal_value

class PUCTPlayer:
    def __init__(self, network, c_puct=1.0):
        """
        PUCT-based player guided by a neural network.

        Args:
            network: GameNetwork-like object with predict(state) -> (policy, value)
            c_puct: exploration constant
        """
        self.network = network
        self.c_puct = c_puct

    def choose_move(self, game, simulations):
        root = PUCTNode(parent=None, move=None, prior=0.0)

        for _ in range(simulations):
            node = root
            scratch = game.clone()

            # 1) SELECTION
            while node.children:
                node = self._select_child_with_tiebreak(node)
                scratch.apply_action(node.move)

            # 2) EXPANSION + EVALUATION
            if not scratch.is_terminal():
                state = encode_state(scratch)

                policy, value = self.network.predict(state)

                mask = legal_action_mask(scratch)
                policy = policy * mask

                if policy.sum() > 0:
                    policy = policy / policy.sum()
                else:
                    # fallback: uniform over legal moves
                    legal = scratch.legal_actions()
                    policy = np.zeros_like(policy)
                    for a in legal:
                        policy[a] = 1 / len(legal)

                for action in scratch.legal_actions():
                    node.children[action] = PUCTNode(
                        parent=node,
                        move=action,
                        prior=policy[action]
                    )
            else:
                # Terminal node
                value = terminal_value(scratch)

            # 3) BACKPROPAGATION
            self._backpropagate(node, value)


        # 4) ACTION SELECTION
        best_action = max(
            root.children.items(),
            key=lambda item: item[1].N
        )[0]

        return best_action
    
    def choose_move_with_pi(self, game, simulations: int):
        """
        Like choose_move(), but also returns pi (visit distribution) for training.

        Returns:
            best_action: int (0..80)
            pi: np.ndarray shape (81,), sums to 1.0
        """
        root = PUCTNode(parent=None, move=None, prior=0.0)

        for _ in range(simulations):
            node = root
            scratch = game.clone()

            # -------- 1) SELECTION --------
            while node.children:
                node = self._select_child_with_tiebreak(node)  # you added this
                scratch.apply_action(node.move)

            # -------- 2) EXPANSION + EVALUATION --------
            if scratch.is_terminal():
                value = terminal_value(scratch)
            else:
                state = encode_state(scratch)
                policy, value = self.network.predict(state)

                mask = legal_action_mask(scratch)
                policy = policy * mask

                s = float(np.sum(policy))
                if s > 0.0:
                    policy = policy / s
                else:
                    policy = np.zeros_like(policy, dtype=np.float32)
                    legal = scratch.legal_actions()
                    for a in legal:
                        policy[a] = 1.0 / len(legal)

                for a in scratch.legal_actions():
                    node.children[a] = PUCTNode(parent=node, move=a, prior=float(policy[a]))

            # -------- 3) BACKPROP --------
            self._backpropagate(node, value)

        if not root.children:
            raise RuntimeError("PUCT root has no children (is the game terminal?)")

        # Build pi from root visit counts
        pi = np.zeros(81, dtype=np.float32)
        total_visits = 0

        for action, child in root.children.items():
            pi[action] = float(child.N)
            total_visits += child.N

        if total_visits > 0:
            pi /= float(total_visits)
        else:
            # extremely unlikely, but keep it safe:
            legal = game.legal_actions()
            for a in legal:
                pi[a] = 1.0 / len(legal)

        best_action = int(np.argmax(pi))
        return best_action, pi



    def _backpropagate(self, node, value):
        """
        Backpropagate the evaluation value up the tree.
        Value is always from the perspective of the current player.
        """
        current = node
        v = value

        while current is not None:
            current.update(v)
            v = -v  # switch perspective
            current = current.parent

    def _terminal_value(self, game, root_player):
        """
        Convert terminal game result to value in [-1, 1].
        """
        if game.game_status == SmallBoard.DRAW:
            return 0.0
        if game.game_status == SmallBoard.X_WIN:
            return 1.0 if root_player == SmallBoard.X else -1.0
        if game.game_status == SmallBoard.O_WIN:
            return 1.0 if root_player == SmallBoard.O else -1.0

    def _select_child_with_tiebreak(self, node):
        best_score = None
        best_children = []

        for child in node.children.values():
            score = child.puct_score(self.c_puct)

            if best_score is None or score > best_score + 1e-12:
                best_score = score
                best_children = [child]
            elif abs(score - best_score) <= 1e-12:
                best_children.append(child)

        return random.choice(best_children)