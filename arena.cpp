#include "arena.hpp"

Arena::Arena (Game* game, NeuralNetwork* a) {
	g = game;
	nn_a = a;
	nn_b = a->clone();
	mcts_a = new MCTS(nn_a);
	mcts_b = new MCTS(nn_b);
}

// using Example = std::tuple<Game*, torch::Tensor, torch::Tensor>;
std::vector<Example> Arena::episode (bool starting_player) {
	Game* curgame = g;
	std::vector<Example> data;
	int step = 0;
	while (true) {
		step++;
		torch::Tensor pi;
		if (starting_player) {
			pi = mcts_a->get_action_prob(curgame);
		} else {
			pi = mcts_b->get_action_prob(curgame);
		}

		data.push_back({
					curgame, pi, torch::tensor({ 0.f }, { torch::kFloat32 })
				});
		int pick = pi.argmax().item<int>();
		Game* next_state = curgame->get_next_state(pick);
		curgame = next_state;
		starting_player = !starting_player;

		if (curgame->is_game_ended() != 0.) {
			break;
		}
	}

	double game_result = curgame->is_game_ended();
	for (size_t i = 0; i < data.size(); i++) {
		if (i % 2 == 0) {
			if (starting_player) {
				// circle = a
				std::get<2>(data[i]) = torch::tensor({ (float)game_result }, { torch::kFloat32 });
			} else {
				// circle = b
				std::get<2>(data[i]) = torch::tensor({ (float)game_result * -1.0 }, { torch::kFloat32 });
			}
		} else {
			if (starting_player) {
				std::get<2>(data[i]) = torch::tensor({ (float)game_result * -1.0 }, { torch::kFloat32 });
			} else {
				std::get<2>(data[i]) = torch::tensor({ (float)game_result }, { torch::kFloat32 });
			}
		}
	}

	return data;
}

void Arena::train (std::vector<Example> examples) {
	delete nn_b;
	nn_b = nn_a->clone();
	nn_a->feed(examples);
	for (size_t i = 1; i < examples.size(); i++) {
		std::cout << std::get<0>(examples[i])->to_string() << " -> " << std::get<2>(examples[i]) << std::endl;
		std::cout << "result : " << std::get<0>(examples[i])->is_game_ended() << std::endl;
		delete std::get<0>(examples[i]);
	}
}

Arena::~Arena () {
	delete nn_b;
	delete nn_a;
	delete g;
}
