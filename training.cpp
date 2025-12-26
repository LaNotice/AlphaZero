#include <torch/torch.h>
#include <iostream>

#include "ttt_game.hpp"
#include "ttt_nn.hpp"
#include "arena.hpp"

int main () {
	auto nn = new TicTacToeNN();
	auto arena = new Arena(nn);

	nn->eval();
	const size_t numEpisodes = 100;
	const size_t numGames = 100;
	for (size_t i = 0; i < numEpisodes; i++) {
		auto g = new TicTacToeState();
		std::vector<Example> exs;
		std::cout << "episode " << i + 1 << "/" << numEpisodes << std::endl;
		for (size_t j = 0; j < numGames; j++) {
			auto local_exs = arena->episode(g, (j % 2 == 1) ? 1 : -1);
			exs.insert(exs.end(), local_exs.begin(), local_exs.end());
			std::cout << "[";
			for (size_t k = 0; k < numGames; k++) {
				if (k < j) { std::cout << "="; }
				else { std::cout << " "; }
			}
			std::cout << "] " << j << "%\r";
			std::cout.flush();
		}
		std::cout << std::endl;
		arena->train(exs);
	}

	torch::serialize::OutputArchive output;
	arena->nn_a->save(output);
	output.save_to("models/ttt.pt");

	delete arena;
}
