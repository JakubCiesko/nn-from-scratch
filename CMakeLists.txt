cmake_minimum_required(VERSION 4.0)
project(project_repo)

set(CMAKE_CXX_STANDARD 20)

add_executable(project_repo
        main.cpp
        src/neural_network/matrix.cpp
        src/neural_network/tensor.cpp
        src/neural_network/optimizer.cpp
        src/data/data_preparator.cpp
        src/neural_network/optimizer.cpp
)
