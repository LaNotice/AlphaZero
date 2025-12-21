#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <torch/torch.h>
#include <string>
#include "game.hpp"

using Example = std::tuple<Game*, torch::Tensor, torch::Tensor>;

struct NeuralNetwork {
	virtual void train (std::vector<Example> examples) = 0;
	virtual std::pair<torch::Tensor, torch::Tensor> predict (Game* g) = 0;
	virtual void save (std::string path) = 0;
	virtual void load (std::string path) = 0;
};

#endif
