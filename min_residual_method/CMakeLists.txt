cmake_minimum_required(VERSION 3.8)
project(projects)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message("OK")
else()
    message("NOT OK")
endif()

set(CUDA_SOURCE_FILES main.cu)

cuda_add_executable(projects ${CUDA_SOURCE_FILES})

target_link_libraries(projects ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})

cuda_add_cublas_to_target(projects)
