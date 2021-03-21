cmake_minimum_required(VERSION 3.17)
project(network_software C)

set(CMAKE_C_STANDARD 11)

add_executable(lab1_server lab1/server.c)
add_executable(lab1_client lab1/client.c)

find_package(Threads REQUIRED)

add_executable(lab2_server lab2/server.c)
target_link_libraries(lab2_server pthread)

add_executable(lab2_client lab2/client.c)

add_executable(lab3_server lab3/server.c)
add_executable(lab3_client lab3/client.c)

add_executable(lab4_server lab4/server.c)
add_executable(lab4_client_tcp lab4/tcp_client.c)
add_executable(lab4_client_udp lab4/udp_client.c)