cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(alphazero)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(alphazero main.cpp)

target_sources(alphazero
	PUBLIC ttt_game.cpp
	PUBLIC ttt_nn.cpp
	PUBLIC ttt_play.cpp
	PUBLIC mcts.cpp
)

target_link_libraries(alphazero "${TORCH_LIBRARIES}")
target_link_libraries(alphazero raylib)
set_property(TARGET alphazero PROPERTY CXX_STANDARD 17)
