#include "ttt_play.hpp"

PlayTTT::PlayTTT () {
	nn = new TicTacToeNN();
	gstate = new TicTacToeState();
	mcts = new MCTS(nn);

	InitWindow(800, 600, "Tic Tac Toe");
	SetTargetFPS(60);
}

void PlayTTT::display () {
	DrawLineEx({ 250, 10 }, { 250, 580 }, 5.f, BLACK);
	DrawLineEx({ 500, 10 }, { 500, 580 }, 5.f, BLACK);
	DrawLineEx({ 10, 200 }, { 780, 200 }, 5.f, BLACK);
	DrawLineEx({ 10, 400 }, { 780, 400 }, 5.f, BLACK);

	for (size_t i = 0; i < gstate->board.size(); i++) {
		const MARK v = gstate->board[i];
		const float x = i % 3;
		const float y = i / 3;
		const Vector2 pos = { x * 250 + 175, y * 200 + 100 };	
		switch (v) {
			case MARK::CIRCLE:
				DrawCircleLines(pos.x, pos.y, 50.0, BLUE);
				break;
			case MARK::CROSS:
				DrawLineEx({
						pos.x - 50,
						pos.y - 50
					}, {
						pos.x + 50,
						pos.y + 50
					}, 5.f, RED);
				DrawLineEx({
						pos.x + 50,
						pos.y - 50
					}, {
						pos.x - 50,
						pos.y + 50
					}, 5.f, RED);
		}
	}
}

void PlayTTT::input () {
	if (IsKeyPressed(KEY_F)) {
		std::cout << gstate->to_string() << std::endl;
		std::cout << gstate << " " << mcts << " " << nn << std::endl;
		auto pi = mcts->get_action_prob(gstate);
		std::cout << pi << std::endl;
		int pick = pi.argmax().item<int>();
		std::cout << "choice is " << pick << std::endl;
		TicTacToeState* ns = dynamic_cast<TicTacToeState*>(gstate->get_next_state(pick));
		delete gstate;
		gstate = ns;
	}

	if (IsKeyPressed(KEY_Q)) {
		// quit
		quit = true;
	}

	const std::array<int, 9> keys = {
		KEY_KP_1, KEY_KP_2, KEY_KP_3, KEY_KP_4, KEY_KP_5,
		KEY_KP_6, KEY_KP_7, KEY_KP_8, KEY_KP_9
	};
	for (size_t i = 0; i < keys.size(); i++) {
		if (IsKeyPressed(keys[i])) {
			TicTacToeState* ns = dynamic_cast<TicTacToeState*>(gstate->get_next_state(i));
			delete gstate;
			gstate = ns;
		}
	}
}


void PlayTTT::run () {
	while (!WindowShouldClose()) {
		BeginDrawing();
			ClearBackground(WHITE);
			display();
			input();
			if (quit) { break; }
		EndDrawing();
	}
	CloseWindow();
}

PlayTTT::~PlayTTT () {
	delete mcts;
	delete nn;
	delete gstate;
}

