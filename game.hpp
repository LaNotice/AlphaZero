#ifndef GAME
#define GAME

#include <cstddef>
#include <vector>
#include <string>

struct Game {
	virtual size_t get_action_size () = 0;
	virtual std::vector<float> get_possible_moves () = 0;
	virtual double is_game_ended () = 0;
	virtual std::string to_string () = 0;
	virtual Game* get_next_state (size_t action) = 0;
};

#endif
