#include <torch/torch.h>
#include <iostream>

#include "ttt_play.hpp"

int main () {
	PlayTTT ttt("models/ttt.pt");
	ttt.run();
}
