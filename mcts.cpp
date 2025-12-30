#include "mcts.hpp"

MCTS::MCTS (NeuralNetwork* nnet) {
	neural_network = nnet;
}

torch::Tensor MCTS::get_action_prob (Game* g, int temp) {
	const int num_sims = 25;

	for (size_t i = 0; i < num_sims; i++) {
		search(g);
	}

	auto possible_actions = g->get_possible_moves();
	torch::Tensor moves = torch::from_blob(
				possible_actions.data(),
				{ 9 },
				torch::kFloat32
			).clone();

	std::string str = g->to_string();
	std::vector<float> counts;
	counts.reserve(g->get_action_size());
	float sum = 0;
	for (size_t i = 0; i < g->get_action_size(); i++) {
		auto pair = std::pair<GameString, size_t>(str, i);
		if (times_edge_visited.find(pair) != times_edge_visited.end()) {
			counts.push_back(times_edge_visited[pair]);
		} else {
			counts.push_back(0);
		}
		sum += counts[i];
	}

	if (temp == 0) {
		size_t max = *std::max_element(counts.begin(), counts.end());
		for (size_t i = 0; i < counts.size(); i++) {
			if (counts[i] != max) {
				counts[i] = 0.f;
			} else {
				counts[i] = 1.f;
			}
		}
		return torch::from_blob(
					counts.data(),
					{ 9 },
					torch::kFloat32
				).clone() * moves;
	}

	sum = 0;
	for (size_t i = 0; i < counts.size(); i++) {
		counts[i] = pow(counts[i], 1. / temp);
		sum += counts[i];
	}

	for (size_t i = 0; i < counts.size(); i++) {
		counts[i] = counts[i] / sum;
	}

	return torch::from_blob(
				counts.data(),
				{ 9 },
				torch::kFloat32
			) * moves;
}

torch::Tensor MCTS::search (Game* g) {
	double cpu_ct = 1.0;
	std::string str = g->to_string();

	if (game_ended.find(str) == game_ended.end()) {
		game_ended[str] = (float)g->is_game_ended();
	}
	
	if (game_ended[str] != 0.f) {
		float value = game_ended[str];
		return torch::tensor({ -value }, { torch::kFloat32 });
	}

	if (policies.find(str) == policies.end()) {
		auto [ pi, v ] = neural_network->predict(g);
		auto possible_moves = g->get_possible_moves();

		auto pmoves = torch::from_blob(
				possible_moves.data(),
				{ 9 },
				torch::kFloat32
			).clone();
		policies[str] = pi * pmoves;
		valid_moves[str] = possible_moves;
		times_state_visited[str] = 0;
		if (policies[str].sum().item<float>() > 0.f) {
			policies[str] /= policies[str].sum();
		} else {
			std::cout << "all legal moves were masked" << std::endl;
		}
		return -v;
	}

	auto all_moves = g->get_possible_moves();
	auto cur_valid_moves = torch::from_blob(
				all_moves.data(),
				{ 9 },
				torch::kFloat32
			).clone();
	float cur_best = -std::numeric_limits<float>::max();
	size_t best_act = 10;

	for (size_t i = 0; i < all_moves.size(); i++) {
		if (all_moves[i] <= 0.f) {
			continue;
		}

		auto pair = std::pair<GameString, size_t>(str, i);
		auto pol_values = policies[str].data_ptr<float>();
		float u = .0f;
		if (q_values.find(pair) != q_values.end()) {
			u = q_values[pair].item<float>() + cpu_ct * pol_values[i] * sqrt(times_state_visited[str]) / (1.0 + times_edge_visited[pair]);
		} else {
			u = cpu_ct * pol_values[i] * sqrt((double)times_state_visited[str] + EPS);
		}

		if (u >= cur_best) {
			cur_best = u;
			best_act = i;
		}
	}
	if (best_act == 10) {
		assert(best_act != 10);
	}
	Game* next = g->get_next_state(best_act);
	auto v = search(next);
	delete next;
	auto pair = std::pair<GameString, size_t>(str, best_act);
	if (q_values.find(pair) != q_values.end()) {
		auto calculation = (times_edge_visited[pair] * q_values[pair] + v) / (times_edge_visited[pair] + 1);
		q_values[pair] = calculation;
		times_edge_visited[pair] += 1;
	} else {
		q_values[pair] = v;
		times_edge_visited[pair] = 1;
	}
	times_state_visited[str] += 1;
	return -v;
}

