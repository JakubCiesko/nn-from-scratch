cmake_minimum_required(VERSION 3.16)
project(neural_network)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# defaults to release build, if not specified, but i specify it in run.sh
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")

# openMP multithreading for fast matrix matmul
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallelization")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


add_executable(
		neural_network
        main.cpp
        src/neural_network/matrix.cpp
        src/neural_network/tensor.cpp
        src/neural_network/optimizer.cpp
        src/data/data_preparator.cpp
        src/neural_network/network.cpp
        src/data/data_preparator.cpp
        src/utils/train.cpp
)
