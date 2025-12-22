#ifndef PLAYTTT_H
#define PLAYTTT_H

#include <torch/torch.h>
#include <raylib.h>

#include "ttt_game.hpp"
#include "ttt_nn.hpp"
#include "play.hpp"

struct PlayTTT : public Play {
	MCTS* mcts;
	TicTacToeState* gstate;
	TicTacToeNN* nn;
	bool quit = false;

	PlayTTT ();

	virtual void display ();
	virtual void input ();
	virtual void run ();

	~PlayTTT ();
};

#endif
