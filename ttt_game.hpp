#ifndef TTT_GAME
#define TTT_GAME

#include <array>
#include <cstddef>
#include <string>

#include "game.hpp"

enum class MARK {
	NONE = 0,
	CIRCLE = 1,
	CROSS = -1
};

struct TicTacToeState : public Game {
	std::array<MARK, 9> board;
	MARK player = MARK::CIRCLE;

	TicTacToeState ();

	virtual size_t get_action_size ();
	virtual std::vector<float> get_possible_moves ();
	virtual double is_game_ended ();
	virtual std::string to_string ();
	virtual Game* get_next_state (size_t action);

	MARK other_player ();
};

#endif
