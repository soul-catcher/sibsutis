cmake_minimum_required(VERSION 3.15)
project(lab5)

set(CMAKE_CXX_STANDARD 17)

find_library(PTHREAD_LIBRARY pthread)

add_executable(lab5 main.cpp)

target_link_libraries(lab5 ${PTHREAD_LIBRARY})
