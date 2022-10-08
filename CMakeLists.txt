cmake_minimum_required(VERSION 3.16)
project(cucounter_C LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "75")

set(SOURCE_DIR "cucounter")
include_directories(${SOURCE_DIR})

set(SOURCES 
  "${SOURCE_DIR}/kernels.cu"
  "${SOURCE_DIR}/hashtable.cu"
)

add_subdirectory(extern/pybind11)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "${SOURCE_DIR}/bindings.cpp")

#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURE "75")
