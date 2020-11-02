cmake_minimum_required(VERSION 3.17)
project(MPT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_executable(lab0_3 lab0/test_lab0_3.cpp)
add_executable(lab1 lab1/TestUFrac.cpp)
add_executable(lab2 lab2/TestTComplex.cpp)
add_executable(lab3 lab3/TestTPNumber.cpp lab3/TPNumber.cpp)
add_executable(lab4 lab4/TestTPNumEditor.cpp lab4/TPNumEditor.cpp)
add_executable(lab5 lab5/TestTFracEditor.cpp lab5/TFracEditor.cpp)
add_executable(lab6 lab6/TestTComplexEditor.cpp lab6/TComplexEditor.cpp)
