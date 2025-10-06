cmake_minimum_required(VERSION 4.0)
project(project_repo)

set(CMAKE_CXX_STANDARD 20)

add_executable(project_repo
        main.cpp
        src/matrix.cpp
        src/tensor.cpp
)
