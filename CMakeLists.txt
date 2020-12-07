cmake_minimum_required (VERSION 3.0)

project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CMakeConfig)

add_subdirectory(src)
