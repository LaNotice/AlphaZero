#include "ttt_game.hpp"

TicTacToeState::TicTacToeState () {
	for (size_t i = 0; i < board.size(); i++) {
		board[i] = MARK::NONE;
	}
}

size_t TicTacToeState::get_action_size () {
	return 3 * 3;
}

std::vector<float> TicTacToeState::get_possible_moves () {
	std::vector<float> moves;
	moves.reserve(9);
	for (size_t i = 0; i < board.size(); i++) {
		if (board[i] != MARK::NONE) {
			moves.push_back(0.f);
		} else {
			moves.push_back(1.f);
		}
	}
	return moves;
}

double TicTacToeState::is_game_ended () {
	const std::array<std::array<size_t, 3>, 8> combinations = {{
		{ 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 },
		{ 0, 3, 6 }, { 1, 4, 7 }, { 2, 5, 8 },
		{ 0, 4, 8 }, { 2, 4, 6 }
	}};

	for (size_t i = 0; i < combinations.size(); i++) {
		auto [ a, b, c ] = combinations[i];
		if (board[a] == MARK::CIRCLE
				&& board[b] == MARK::CIRCLE
				&& board[c] == MARK::CIRCLE) {
			return 1.0;
		}
		if (board[a] == MARK::CROSS
				&& board[b] == MARK::CROSS
				&& board[c] == MARK::CROSS) {
			return -1.0;
		}
	}

	bool moves_left = false;
	for (size_t i = 0; i < board.size(); i++) {
		if (board[i] == MARK::NONE) { moves_left = true; break; }
	}

	if (moves_left) {
		return 0.0;
	}

	return 1e-4;
}

std::string TicTacToeState::to_string () {
	std::string str = "";
	for (size_t i = 0; i < board.size(); i++) {
		if (board[i] == MARK::CROSS) {
			str += 'X';
		} else if (board[i] == MARK::CIRCLE) {
			str += 'O';
		} else {
			str += ' ';
		}
		str += "|";
		if (i % 3 == 2) { str += '\n'; }
	}
	return str;
}

Game* TicTacToeState::get_next_state (size_t action) {
	assert(board[action] == MARK::NONE);
	TicTacToeState* new_state = new TicTacToeState();
	new_state->board = board;
	new_state->board[action] = player;
	new_state->player = other_player();
	return new_state;
}

MARK TicTacToeState::other_player () {
	if (player == MARK::CIRCLE) {
		return MARK::CROSS;
	}
	return MARK::CIRCLE;
}

