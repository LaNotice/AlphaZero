#include <torch/torch.h>
#include <iostream>

#include "ttt_play.hpp"

int main () {
	PlayTTT pt = PlayTTT();
	pt.run();
}
