#ifndef ARENA_H
#define ARENA_H

#include <vector>

#include "game.hpp"
#include "neural_network.hpp"
#include "mcts.hpp"

struct Arena {
	NeuralNetwork* nn_b;
	NeuralNetwork* nn_a;
	MCTS* mcts_a;
	MCTS* mcts_b;

	Arena (NeuralNetwork* a);
	std::vector<Example> episode (Game* g, int starting_player = 1);
	void train (std::vector<Example> data);
	~Arena();
};

#endif
