#include <torch/torch.h>
#include <iostream>

#include "ttt_game.hpp"
#include "ttt_nn.hpp"
#include "mcts.hpp"

int main () {
	NeuralNetwork* nn = new TicTacToeNN();
	Game* game = new TicTacToeState();

	MCTS mcts = MCTS(game, nn);
	auto v = mcts.search(game);
	std::cout << "value " << v << std::endl;

	auto pi = mcts.get_action_prob(game, 1);
	std::cout << "policy " << pi << std::endl;
}
