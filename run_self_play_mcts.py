from self_play_mcts import MCTSSelfPlay, save_dataset

if __name__ == "__main__":
    generator = MCTSSelfPlay(iterations=1000)
    dataset = generator.generate(num_games=1000)
    save_dataset(dataset, "mcts_pretrain_data.npz")
