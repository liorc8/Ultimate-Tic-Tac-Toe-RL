from src.self_play_mcts import MCTSSelfPlay, save_dataset

if __name__ == "__main__":
    generator = MCTSSelfPlay(iterations=200)
    dataset = generator.generate(num_games=100)
    save_dataset(dataset, "mcts_pretrain_data.npz")
