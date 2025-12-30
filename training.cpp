#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <future>
#include <iostream>

#include "ttt_game.hpp"
#include "ttt_nn.hpp"
#include "arena.hpp"

int main () {
	auto nn = new TicTacToeNN();
	auto arena = new Arena(nn);
	auto rng = std::default_random_engine {};

	nn->eval();
	const size_t numEpisodes = 100;
	const size_t numGames = 100;
	for (size_t i = 0; i < numEpisodes; i++) {
		auto g = new TicTacToeState();
		std::vector<Example> exs;
		std::array<std::future<std::vector<Example>>, numGames> futures;
		std::cout << "episode " << i + 1 << "/" << numEpisodes << std::endl;
		for (size_t j = 0; j < numGames; j++) {
			futures[j] = std::async(&Arena::episode, arena, g, (j % 2 == 0) ? 1 : -1);
		}
		for (size_t j = 0; j < numGames; j++) {
			futures[j].wait();
			auto local_exs = futures[j].get();
			exs.insert(exs.end(), local_exs.begin(), local_exs.end());
			std::cout << "[";
			for (size_t k = 0; k <= numGames; k++) {
				if (k <= j) { std::cout << "="; }
				else { std::cout << " "; }
			}
			std::cout << "] " << ((float)(j + 1) / (float)numGames) * 100 << "%\r";
			std::cout.flush();
		}
		std::cout << std::endl;
		std::shuffle(std::begin(exs), std::end(exs), rng);
		arena->train(exs);
	}

	torch::serialize::OutputArchive output;
	if (arena->last_train == 'A') {
		arena->nn_a->save(output);
	} else {
		arena->nn_b->save(output);
	}
	output.save_to("models/ttt.pt");

	delete arena;
}
