#ifndef MCTS_H
#define MCTS_H

#include <torch/torch.h>
#include <map>
#include <vector>
#include <utility>
#include <cstddef>

#include "game.hpp"
#include "neural_network.hpp"

using GameString = std::string;

const double EPS = 1e-8;

struct MCTS {
	std::map<std::pair<GameString, size_t>, torch::Tensor> q_values;
	std::map<std::pair<GameString, size_t>, int> times_edge_visited;
	std::map<GameString, int> times_state_visited;
	std::map<GameString, double> game_ended; 
	std::map<GameString, std::vector<float>> valid_moves;
	std::map<GameString, torch::Tensor> policies;
	NeuralNetwork* neural_network = nullptr;

	MCTS (NeuralNetwork* nnet);
	torch::Tensor get_action_prob (Game* g, int temp = 1);
	torch::Tensor search (Game* g);
	~MCTS ();
};

#endif
