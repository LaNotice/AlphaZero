#include <torch/torch.h>
#include <iostream>

#include "ttt_game.hpp"
#include "ttt_nn.hpp"
#include "arena.hpp"

int main () {
	auto g = new TicTacToeState();
	auto nn = new TicTacToeNN();
	auto arena = new Arena(g, nn);

	const size_t numEpisodes = 10;
	for (size_t i = 0; i < numEpisodes; i++) {
		auto exs = arena->episode(i % 2 == 0);
		arena->train(exs);
		std::cout << "episode " << i + 1 << "/" << numEpisodes << std::endl;
		break;
	}

	torch::serialize::OutputArchive output;
	arena->nn_a->save(output);
	output.save_to("models/ttt_1000.pt");

	delete arena;
}
