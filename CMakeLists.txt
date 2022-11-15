cmake_minimum_required(VERSION 3.16)
project(cucounter_C LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "-O3")

set(SOURCE_DIR "cucounter/backend")
include_directories(${SOURCE_DIR})

set(SOURCES 
  "${SOURCE_DIR}/kernels.cu"
  "${SOURCE_DIR}/hashtable.cu"
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
