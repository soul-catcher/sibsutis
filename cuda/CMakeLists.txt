cmake_minimum_required(VERSION 3.16)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab1 lab01.cu)
add_executable(lab2 lab02.cu)
add_executable(lab3 lab03.cu)
add_executable(lab4 lab04.cu)
add_executable(lab6 lab06.cu)
add_executable(lab6-2 lab06-2.cu)
add_executable(lab7 lab07.cu)

set_target_properties(
        lab1 lab2 lab3 lab4 lab6 lab6-2 lab7
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
