cmake_minimum_required(VERSION 3.19)
project(test1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

add_executable(test1 main.cpp barrier.cpp barrier.h)