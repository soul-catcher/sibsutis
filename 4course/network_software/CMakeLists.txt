cmake_minimum_required(VERSION 3.17)
project(network_software C)

set(CMAKE_C_STANDARD 11)

add_executable(lab1_server lab1/server.c)
add_executable(lab1_client lab1/client.c)