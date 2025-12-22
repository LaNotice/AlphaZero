#ifndef PLAY_H
#define PLAY_H

#include "game.hpp"
#include "neural_network.hpp"
#include "mcts.hpp"

struct Play {
	virtual void display () = 0;
	virtual void input () = 0;
	virtual void run () = 0;
};

#endif
