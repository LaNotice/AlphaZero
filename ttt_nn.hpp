#ifndef TTT_NN
#define TTT_NN

#include <torch/torch.h>
#include <cassert>
#include <iostream>
#include "neural_network.hpp"
#include "ttt_game.hpp"

struct TicTacToeNN : public NeuralNetwork {
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};

	torch::nn::Linear fc1{nullptr}, fc2{nullptr};

	torch::nn::LayerNorm ln_fc1{nullptr}, ln_fc2{nullptr};

	torch::nn::Dropout dropout{nullptr};
	torch::nn::Linear pi_head{nullptr}, v_head{nullptr};

	TicTacToeNN ();

	virtual void feed (std::vector<Example> examples);
	virtual std::pair<torch::Tensor, torch::Tensor> predict (Game* g);
	TicTacToeNN* clone () const;
};

#endif
