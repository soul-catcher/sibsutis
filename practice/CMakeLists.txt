cmake_minimum_required(VERSION 3.14)
project(practice C)

set(CMAKE_C_STANDARD 11)

find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIR})

add_executable(practice src/main.c src/set.c src/set.h)

target_link_libraries(practice ${CURSES_LIBRARIES})

add_executable(tests test/main.c src/set.c src/set.h)
target_link_libraries(tests criterion)