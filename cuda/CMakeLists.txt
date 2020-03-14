cmake_minimum_required(VERSION 3.16)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab1 lab01.cu)
add_executable(lab2 lab02.cu)
add_executable(lab3 lab03.cu)

set_target_properties(
        lab1 lab2 lab3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
