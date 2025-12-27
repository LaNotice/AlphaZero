#include "ttt_nn.hpp"

TicTacToeNN::TicTacToeNN () {
	const int n_chan = 3;
	const double drop = 0.3;
	const int action_size = 9;
	const int flat_size = action_size * n_chan;

	conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, n_chan, 3).padding(1)));
	bn1 = register_module("bn1", torch::nn::BatchNorm2d(n_chan));

	conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(n_chan, n_chan, 3).padding(1)));
	bn2 = register_module("bn2", torch::nn::BatchNorm2d(n_chan));

	conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(n_chan, n_chan, 3).padding(1)));
	bn3 = register_module("bn3", torch::nn::BatchNorm2d(n_chan));

	conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(n_chan, n_chan, 3).padding(1)));
	bn4 = register_module("bn4", torch::nn::BatchNorm2d(n_chan));

	fc1 = register_module("fc1", torch::nn::Linear(flat_size, 1024));
	ln_fc1 = register_module("ln_fc1",
				torch::nn::LayerNorm(torch::nn::LayerNormOptions({ 1024 }))
			);

	fc2 = register_module("fc2", torch::nn::Linear(1024, 512));
	ln_fc2 = register_module("ln_fc2",
				torch::nn::LayerNorm(torch::nn::LayerNormOptions({ 512 }))
			);

	dropout = register_module("dropout",
				torch::nn::Dropout(torch::nn::DropoutOptions(drop))
			);

	pi_head = register_module("pi", torch::nn::Linear(512, action_size));
	v_head = register_module("v", torch::nn::Linear(512, 1));
}

void TicTacToeNN::feed (std::vector<Example> examples) {
	const double learning_rate = 0.001;

	torch::optim::Adam optimizer(
		this->parameters(),
		torch::optim::AdamOptions(learning_rate)
	);

	for (size_t i = 0; i < examples.size(); i++) {
		auto [ game, player, target_pi, target_v ] = examples[i];
		auto [ pred_pi, pred_v ] = predict(game);

		this->train();
		auto l_pi = -(target_pi * torch::log(pred_pi + 1e-8)).sum(1).mean();
		auto l_v = torch::mse_loss(pred_v, target_v);

		auto loss = l_pi + l_v;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
	}
}

std::pair<torch::Tensor, torch::Tensor> TicTacToeNN::predict (Game* abstract) {
	this->eval();

	TicTacToeState* g = (TicTacToeState*)abstract;
	std::array<float, 2 * 9> board_copy;
	for (size_t i = 0; i < 9; i++) {
		board_copy[i] = static_cast<float>(g->board[i]);
	}
	for (size_t i = 0; i < 9; i++) {
		board_copy[i + 9] = static_cast<float>(g->player);
	}
	auto x = torch::from_blob(
				board_copy.data(),
				{ 1, 2, 3, 3 },
				torch::kFloat32
			).clone();

	if (!torch::isfinite(x).all().item<bool>()) {
		for (size_t i = 0; i < g->board.size(); i++) {
			std::cout << (float)g->board[i] << std::endl;
		}
		std::cout << x << std::endl;
		assert(torch::isfinite(x).all().item<bool>());
	}

	x = torch::relu(bn1(conv1(x)));
	x = torch::relu(bn2(conv2(x)));
	x = torch::relu(bn3(conv3(x)));
	x = torch::relu(bn4(conv4(x)));

	x = x.view({ x.size(0), -1 });

	x = dropout(torch::relu(ln_fc1(fc1(x))));
	x = dropout(torch::relu(ln_fc2(fc2(x))));

	torch::Tensor pi = torch::softmax(pi_head(x), 1);
	torch::Tensor v = torch::tanh(v_head(x));

	return { pi, v };
}

TicTacToeNN* TicTacToeNN::clone () const {
	return new TicTacToeNN(*this);
}

