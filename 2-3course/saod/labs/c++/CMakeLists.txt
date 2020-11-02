cmake_minimum_required(VERSION 3.12)
project(saod)

set(CMAKE_CXX_STANDARD 17)
#find_package(SFML 2.5 COMPONENTS window graphics REQUIRED)

add_executable(lab_7 lab_7.cpp)
add_executable(lab_9 lab_9.cpp)
add_executable(mur mur.cpp)
add_executable(haffman haffman.cpp)
add_executable(shannon shannon.cpp)
add_executable(fano fano.cpp)

#target_link_libraries(saod sfml-window sfml-graphics)
