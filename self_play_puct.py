import numpy as np
from ultimate_board import UltimateBoard
from small_board import SmallBoard

from encoder import encode_state
from value_target import terminal_value
from puct_player import PUCTPlayer
from game_network import GameNetwork


class SelfPlayGenerator:
    def __init__(self, simulations=50, c_puct=1.0):
        self.simulations = simulations
        self.network = GameNetwork()  # Dummy network
        self.player = PUCTPlayer(self.network, c_puct=c_puct)

    def play_one_game(self):
        """
        Play one self-play game using PUCT.

        Returns:
            list of (state, pi, z)
        """
        game = UltimateBoard()
        memory = []

        while not game.is_terminal():
            state = encode_state(game)
            action, pi = self.player.choose_move_with_pi(
                game, self.simulations
            )

            # store (state, pi, current_player)
            memory.append((state, pi, game.current_player))

            game.apply_action(action)

        # Game ended → compute final value
        z_final = terminal_value(game)

        # Assign z to all states from the correct perspective
        dataset = []
        for state, pi, player_at_state in memory:
            z = float(game.value_for(player_at_state))
            dataset.append((state, pi, z))
        return dataset


    def generate(self, num_games):
        """
        Generate dataset from multiple self-play games.
        """
        data = []
        for i in range(num_games):
            game_data = self.play_one_game()
            data.extend(game_data)
            print(f"Game {i+1}/{num_games} finished, samples: {len(game_data)}")

        return data


def save_dataset(dataset, path):
    """
    Save dataset to disk using numpy.
    """
    states = np.array([d[0] for d in dataset], dtype=np.float32)
    pis = np.array([d[1] for d in dataset], dtype=np.float32)
    zs = np.array([d[2] for d in dataset], dtype=np.float32)

    np.savez_compressed(path, states=states, pis=pis, zs=zs)
    print(f"Dataset saved to {path}")
