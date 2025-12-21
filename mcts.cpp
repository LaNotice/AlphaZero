#include "mcts.hpp"

MCTS::MCTS (Game* g, NeuralNetwork* nnet) {
	initial = g;
	neural_network = nnet;
}

torch::Tensor MCTS::get_action_prob (Game* g, int temp) {
	const int num_sims = 3;

	for (size_t i = 0; i < num_sims; i++) {
		search(g);
	}

	// weighted actions
	std::string str = g->to_string();
	std::vector<float> counts;
	counts.reserve(initial->get_action_size());
	double sum = 0;
	for (size_t i = 0; i < initial->get_action_size(); i++) {
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
				counts[i] = 0;
			} else {
				counts[i] = 1;
			}
		}
		return torch::from_blob(
					counts.data(),
					{ 9 },
					torch::kFloat32
				).clone();
	}

	sum = 0;
	for (size_t i = 0; i < counts.size(); i++) {
		counts[i] = pow(counts[i], 1. / temp);
		sum += counts[i];
	}

	return torch::from_blob(
				counts.data(),
				{ 9 },
				torch::kFloat32
			);
}

torch::Tensor MCTS::search (Game* g) {
	double cpu_ct = 1;
	std::string str = g->to_string();

	if (game_ended.find(str) == game_ended.end()) {
		game_ended[str] = g->is_game_ended();
	} else if (game_ended[str]) {
		float value = game_ended[str];
		return torch::tensor({ value }, { torch::kFloat32 });
	}

	if (policies.find(str) == policies.end()) {
		// leaf node
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
		return -v;
	}

	auto all_moves = g->get_possible_moves();
	auto cur_valid_moves = torch::from_blob(
				valid_moves[str].data(),
				{ 9 },
				torch::kFloat32
			).clone();
	float cur_best = -99999.9f;
	size_t best_act = 0;

	auto data = cur_valid_moves.data_ptr<float>();
	auto n = cur_valid_moves.numel();
	for (size_t i = 0; i < n; i++) {
		if (data[i] <= 0) {
			continue;
		}

		auto pair = std::pair<GameString, size_t>(str, i);
		auto pol_values = policies[str].data_ptr<float>();
		double u = .0f;
		if (q_values.find(pair) != q_values.end()) {
			u = q_values[pair].item<float>() * cpu_ct * pol_values[i] * sqrt(times_state_visited[str]) / (1.0 + times_edge_visited[pair]);
		} else {
			u = cpu_ct * pol_values[i] * sqrt((double)times_state_visited[str] + EPS);
		}

		if (u > cur_best) {
			cur_best = u;
			best_act = i;
		}
	}

	Game* next = initial->get_next_state(best_act);
	auto v = search(next);
	auto pair = std::pair<GameString, size_t>(str, best_act);
	if (q_values.find(pair) != q_values.end()) {
		q_values[pair] = (times_edge_visited[pair] * q_values[pair] + v) / (times_edge_visited[pair] + 1);
		times_edge_visited[pair] += 1;
	} else {
		q_values[pair] = v;
		times_edge_visited[pair] = 1;
	}
	return -v;
}

