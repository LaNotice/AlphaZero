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
		int temp = 1;
		if (starting_player == 1) {
			pi = mcts_a->get_action_prob(game, temp);
		} else {
			pi = mcts_b->get_action_prob(game, temp);
		}

		data.push_back({
					game, starting_player, pi, torch::tensor({ 0 }, { torch::kFloat32 })
				});
		int pick = torch::multinomial(pi, 1).item<int>();
		Game* next_state = game->get_next_state(pick);
		game = next_state;
		starting_player *= -1;
		if (game->is_game_ended() != 0.0) {
			break;
		}
	}
	double game_result = game->is_game_ended();
	if (game_result == 0.0 || game_result == 1e-4) {
		draws++;
	} else if (game_result == -1.0) {
		if (original_starter == 1) { bwins++; }
		else { awins++; }
		cross_win++;
	} else if (game_result == 1.0) {
		if (original_starter == 1) { awins++; } else { bwins++; }
		circle_win++;
	}
	for (size_t i = 0; i < data.size(); i++) {
		float z = ((int)game_result == std::get<1>(data[i])) ? 1.f : -1.f;
		if (game_result == 0.0 || game_result == 1e-4) {
			z = 0.f;
		}
		auto new_v = torch::tensor({ z }, { torch::kFloat32 });
		std::get<3>(data[i]) = new_v;
	}
	return data;
}

void Arena::train (std::vector<Example> examples) {
	float total = awins + bwins + draws;
	if (awins > bwins) {
		// a better
		std::cout << "A better by " << (float)awins / (float)bwins << std::endl;
		delete nn_b;
		nn_b = nn_a->clone();
		nn_a->feed(examples);
		last_train = 'A';
	} else if (bwins > awins) {
		// b better
		std::cout << "B better by " << (float)bwins / (float)awins << std::endl;
		delete nn_a;
		nn_a = nn_b->clone();
		nn_b->feed(examples);
		last_train = 'B';
	} else {
		std::cout << "training rejected (" << awins << "/" << bwins << "/" << draws << ")" << std::endl;
		// equals
		if (last_train == 'A') {
			delete nn_a;
			nn_a = nn_b->clone();
		} else {
			delete nn_b;
			nn_b = nn_a->clone();
		}
		nn_a->feed(examples);
		last_train = 'A';
	}
	std::cout << "X wins " << cross_win << ", O wins " << circle_win << std::endl;
	cross_win = 0;
	circle_win = 0;
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
	awins = 0;
	bwins = 0;
	draws = 0;
}

Arena::~Arena () {
	delete nn_b;
	delete nn_a;
	delete mcts_a;
	delete mcts_b;
}
