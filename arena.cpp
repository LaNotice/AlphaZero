#include "arena.hpp"

Arena::Arena (NeuralNetwork* a) {
	nn_a = a;
	nn_b = a->clone();
	mcts_a = new MCTS(nn_a);
	mcts_b = new MCTS(nn_b);
}

std::vector<Example> Arena::episode (Game* game, int starting_player) {
	std::vector<Example> data;
	int original_starter = starting_player;
	int step = 0;
	while (true) {
		step++;
		torch::Tensor pi;
		if (starting_player == 1) {
			pi = mcts_a->get_action_prob(game);
		} else {
			pi = mcts_b->get_action_prob(game);
		}

		data.push_back({
					game, pi, torch::tensor({ 0.f }, { torch::kFloat32 })
				});
		int pick = pi.argmax().item<int>();
		Game* next_state = game->get_next_state(pick);
		game = next_state;
		starting_player *= -1;

		if (game->is_game_ended() != 0.0) {
			break;
		}
	}

	double game_result = game->is_game_ended() * starting_player;
	for (size_t i = 0; i < data.size(); i++) {
		float outcome = game_result * pow(-1, (i % 2 == 0) * starting_player);
		auto new_v = torch::tensor({ outcome }, { torch::kFloat32 });
		std::get<2>(data[i]) = new_v;
	}

	return data;
}

void Arena::train (std::vector<Example> examples) {
	delete nn_b;
	nn_b = nn_a->clone();
	nn_a->feed(examples);
	std::vector<Game*> deleted;
	deleted.reserve(examples.size());
	for (size_t i = 0; i < examples.size(); i++) {
		auto ptr = std::get<0>(examples[i]);
		if (std::find(deleted.begin(), deleted.end(), ptr) == deleted.end()) {
			deleted.push_back(ptr);
			delete ptr;
		}
	}
	delete mcts_a;
	delete mcts_b;
	mcts_a = new MCTS(nn_a);
	mcts_b = new MCTS(nn_b);
}

Arena::~Arena () {
	delete nn_b;
	delete nn_a;
	delete mcts_a;
	delete mcts_b;
}
