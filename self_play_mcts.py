import numpy as np
from ultimate_board import UltimateBoard
from mcts import MCTSPlayer
from encoder import encode_state


class MCTSSelfPlay:
    def __init__(self, iterations=200):
        self.iterations = iterations
        self.player = MCTSPlayer()

    def play_one_game(self):
        game = UltimateBoard()
        memory = []

        while not game.is_terminal():
            state = encode_state(game)
            action, pi = self.player.choose_move_with_pi(game, self.iterations)

            memory.append((state, pi, game.current_player))
            game.apply_action(action)

        dataset = []
        for state, pi, player_at_state in memory:
            z = float(game.value_for(player_at_state))
            dataset.append((state, pi, z))

        return dataset

    def generate(self, num_games):
        data = []
        for i in range(num_games):
            game_data = self.play_one_game()
            data.extend(game_data)
            print(f"Game {i+1}/{num_games} finished, samples: {len(game_data)}")
        return data


def save_dataset(dataset, path):
    states = np.array([d[0] for d in dataset], dtype=np.float32)
    pis = np.array([d[1] for d in dataset], dtype=np.float32)
    zs = np.array([d[2] for d in dataset], dtype=np.float32)

    np.savez_compressed(path, states=states, pis=pis, zs=zs)
    print(f"Dataset saved to {path}")
