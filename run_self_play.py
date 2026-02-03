from self_play import SelfPlayGenerator, save_dataset

if __name__ == "__main__":
    generator = SelfPlayGenerator(
        simulations=50,
        c_puct=1.0
    )

    dataset = generator.generate(num_games=10)
    save_dataset(dataset, "self_play_data.npz")
