cmake_minimum_required(VERSION 3.28)
project(g4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(critCore CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

file(GLOB G4_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(g4 SHARED ${G4_SOURCES})
target_include_directories(g4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(g4 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -O3 -Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_link_libraries(g4 PRIVATE crit::core crit::cuda CUDA::cudart CUDA::cublasLt)
set_target_properties(g4 PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
