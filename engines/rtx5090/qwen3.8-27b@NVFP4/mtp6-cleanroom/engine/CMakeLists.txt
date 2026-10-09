cmake_minimum_required(VERSION 3.28)
project(q38_cleanroom LANGUAGES CXX CUDA)
# Clean-room Qwen3.8-27B NVFP4 engine on crit Core (CUDA backend, copied unmodified
# under core/) and the crit Forge device-speculative-round controller (src/forge/).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CRIT_CORE_CUDA ON CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/core" "${CMAKE_CURRENT_BINARY_DIR}/crit-core")
find_package(CUDAToolkit 13.1 REQUIRED)

set(Q38_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_library(q38engine STATIC ${Q38_SRC}/kernels.cu ${Q38_SRC}/model.cu ${Q38_SRC}/engine.cpp
            ${Q38_SRC}/forge/device_round.cpp ${Q38_SRC}/forge/greedy_accept.cu)
target_include_directories(q38engine PUBLIC ${Q38_SRC} ${Q38_SRC}/forge)
target_link_libraries(q38engine PUBLIC crit::cuda CUDA::cudart CUDA::cublas)
target_compile_options(q38engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)

add_library(q38 SHARED ${Q38_SRC}/capi.cpp)
target_link_libraries(q38 PRIVATE q38engine)

add_executable(q38bench ${Q38_SRC}/bench.cpp)
target_link_libraries(q38bench PRIVATE q38engine)
