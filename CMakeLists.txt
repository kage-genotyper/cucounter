cmake_minimum_required(VERSION 3.16)
project(cucounter_C LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "-O3")

set(SOURCE_DIR "cucounter/backend")
include_directories(${SOURCE_DIR})
include_directories("/usr/local/cuda-11.4/targets/x86_64-linux/include/")


set(SOURCES 
  "${SOURCE_DIR}/kernels.cu"
  "${SOURCE_DIR}/hashtable.cu"
)

if(${CUDA_ERROR_CHECK} MATCHES "True")
  add_compile_definitions(_CUDA_ERROR_CHECK)
endif()
if(${USE_COOPERATIVE_GROUPS} MATCHES "True")
  add_compile_definitions(_USE_COOPERATIVE_GROUPS)
endif()
if(DEFINED COOPERATIVE_GROUP_SIZE)
  add_compile_definitions(_COOPERATIVE_GROUP_SIZE=${COOPERATIVE_GROUP_SIZE})
endif()
if(${USE_MURMUR_HASH} MATCHES "True")
  add_compile_definitions(_USE_MURMUR_HASH)
endif()
if(${USE_AOS} MATCHES "True")
  add_compile_definitions(_USE_AOS)
endif()

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
