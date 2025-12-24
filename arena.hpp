#ifndef ARENA_H
#define ARENA_H

#include <vector>

#include "game.hpp"
#include "neural_network.hpp"
#include "mcts.hpp"

struct Arena {
	Game* g;
	NeuralNetwork* nn_b;
	NeuralNetwork* nn_a;
	MCTS* mcts_a;
	MCTS* mcts_b;

	Arena (Game* g, NeuralNetwork* a);
	std::vector<Example> episode (int starting_player = 1);
	void train (std::vector<Example> data);
	~Arena();
};

#endif
