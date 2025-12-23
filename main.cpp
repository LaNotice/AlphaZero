#include <torch/torch.h>
#include <iostream>

#include "ttt_play.hpp"

int main () {
	PlayTTT ttt("models/ttt_1000.pt");
	ttt.run();
}
