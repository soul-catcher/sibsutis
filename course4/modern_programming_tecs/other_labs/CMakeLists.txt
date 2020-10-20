cmake_minimum_required(VERSION 3.17)
project(MPT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_executable(Lab0_3 lab0/test_lab0_3.cpp)
add_executable(lab1 lab1/TestUFrac.cpp)
add_executable(lab2 lab2/TestTComplex.cpp)

