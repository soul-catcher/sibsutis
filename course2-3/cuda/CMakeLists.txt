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
add_executable(lab8-1 lab08-1.cu)
add_executable(lab8-2 lab08-2.cu)
add_executable(course_cuda course_work/cuda.cu)
add_executable(course_trust course_work/trust.cu)
add_executable(course_cublas course_work/cublas.cu)

target_link_libraries(lab8-1 -lcublas)
target_link_libraries(lab8-2 -lcufft)
target_link_libraries(course_cublas -lcublas)

set_target_properties(
        lab1 lab2 lab3 lab4 lab6 lab6-2 lab7 lab8-1 lab8-2 course_cuda course_trust course_cublas
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
