cmake_minimum_required(VERSION 3.8)
project(projects)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA)

if(CUDA_FOUND)
    message("OK")
else()
    message("NOT OK")
endif()

set(CUDA_SOURCE_FILES main.cu)
cuda_add_executable(projects ${CUDA_SOURCE_FILES})
