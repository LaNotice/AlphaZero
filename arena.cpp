#include "arena.hpp"

Arena::Arena (Game* game, NeuralNetwork* a) {
	g = game;
	nn_a = a;
	nn_b = a->clone();
	mcts_a = new MCTS(nn_a);
	mcts_b = new MCTS(nn_b);
}

std::vector<Example> Arena::episode (int starting_player) {
	Game* curgame = g;
	std::vector<Example> data;
	int original_starter = starting_player;
	int step = 0;
	while (true) {
		step++;
		torch::Tensor pi;
		if (starting_player == 1) {
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
		starting_player *= starting_player;

		if (curgame->is_game_ended() != 0.0) {
			break;
		}
	}

//	std::cout << "FINAL STATE" << std::endl;
//	std::cout << curgame->to_string() << std::endl;
//	std::cout << "===============" << std::endl;

	double game_result = curgame->is_game_ended() * starting_player;
	for (size_t i = 0; i < data.size(); i++) {
		double perspective = 0.0;
		if (original_starter == 1) {
			perspective = (i % 2 == 0) ? 1.0 : -1.0;
		} else {
			perspective = (i % 2 == 0) ? -1.0: 1.0;
		}
		float outcome = perspective * game_result;
		// float v = game_result * pow(-1, starting_player * (i % 2 == 0));
		auto new_v = torch::tensor({ outcome }, { torch::kFloat32 });
		std::get<2>(data[i]) = new_v;
//		std::cout << std::get<0>(data[i])->to_string() << std::endl;	
//		std::cout << "value " << v << "/" << new_v << std::endl;
//		std::cout << "vs result " << game_result << std::endl;
	}

	return data;
}

void Arena::train (std::vector<Example> examples) {
	delete nn_b;
	nn_b = nn_a->clone();
	nn_a->feed(examples);
	for (size_t i = 1; i < examples.size(); i++) {
		delete std::get<0>(examples[i]);
	}
	delete mcts_a;
	delete mcts_b;
	mcts_a = new MCTS(nn_a);
	mcts_b = new MCTS(nn_b);
}

Arena::~Arena () {
	delete nn_b;
	delete nn_a;
	delete g;
	delete mcts_a;
	delete mcts_b;
}
