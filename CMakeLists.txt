cmake_minimum_required(VERSION 3.16)
project(cucounter_backend LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "-O3")

set(SOURCE_DIR "cucounter/backend")
include_directories(${SOURCE_DIR})
include_directories("${CUDA_INCLUDE_DIRS}")
#include_directories("/usr/local/cuda-11.4/targets/x86_64-linux/include/")


set(SOURCES 
  "${SOURCE_DIR}/kernels.cu"
  "${SOURCE_DIR}/hashtable.cu"
)

if(${CUDA_ERROR_CHECK} MATCHES "True")
  add_compile_definitions(__CUDA_ERROR_CHECK__)
endif()
if(${USE_COOPERATIVE_GROUPS} MATCHES "True")
  add_compile_definitions(__USE_COOPERATIVE_GROUPS__)
endif()
if(DEFINED COOPERATIVE_GROUP_SIZE)
  add_compile_definitions(__COOPERATIVE_GROUP_SIZE__=${COOPERATIVE_GROUP_SIZE})
endif()
if(${USE_MURMUR_HASH} MATCHES "True")
  add_compile_definitions(__USE_MURMUR_HASH__)
endif()
if(${USE_AOS} MATCHES "True")
  add_compile_definitions(__USE_AOS__)
endif()

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
