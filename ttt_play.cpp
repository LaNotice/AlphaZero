#include "ttt_play.hpp"

PlayTTT::PlayTTT (std::string model_path) {
	nn = new TicTacToeNN();
	torch::serialize::InputArchive input;
	if (model_path != "") {
		input.load_from(model_path);
		nn->load(input);
	}

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
				DrawCircle(pos.x, pos.y, 50.0, BLUE); 
				DrawCircle(pos.x, pos.y, 45.0, WHITE); 
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
		auto pi = mcts->get_action_prob(gstate, 1);
		int pick = pi.argmax().item<int>();
		std::cout << "policy : " << pi << std::endl;
		TicTacToeState* ns = dynamic_cast<TicTacToeState*>(gstate->get_next_state(pick));
		delete gstate;
		gstate = ns;
	}

	if (IsKeyPressed(KEY_V)) {
		auto [pi, v] = nn->predict(gstate);
		std::cout << "board state\n" << gstate->to_string() << std::endl;
		std::cout << "policy vector " << pi << std::endl;
		std::cout << "value vector " << v << std::endl;
		std::cout << "OPTIONS" << std::endl;
		auto moves = gstate->get_possible_moves();
		for (size_t i = 0; i < moves.size(); i++) {
			if (moves[i] > 0.f) {
				TicTacToeState* simulation = dynamic_cast<TicTacToeState*>(gstate->get_next_state(i));
				auto [ spi, sv ] = nn->predict(simulation);
				std::cout << "move " << i + 1 << ": value " << sv << std::endl;
				delete simulation;
			} else {
				std::cout << "move " << i + 1 << ": IMPOSSIBLE" << std::endl;
			}
		}
	}
	if (IsKeyPressed(KEY_R)) {
		// reset
		delete gstate;
		delete mcts;
		gstate = new TicTacToeState();
		mcts = new MCTS(nn);
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

