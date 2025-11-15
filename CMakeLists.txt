cmake_minimum_required(VERSION 3.16)
project(project_repo)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Default to Release build if not specified ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- Compiler optimizations for speed ---
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
#
## --- Optional: Enable multithreading (OpenMP) ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallelization")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
#

add_executable(project_repo
        main.cpp
        src/neural_network/matrix.cpp
        src/neural_network/tensor.cpp
        src/neural_network/optimizer.cpp
        src/data/data_preparator.cpp
        src/neural_network/network.cpp
)
