cmake_minimum_required(VERSION 3.28)
project(fnl LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120a)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(critCore 0.1.0 EXACT CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(FNL_SOURCES src/gemv.cu src/gemm.cu src/prefill.cu src/prefill_net.cu src/experts.cu src/layers.cu src/model.cu src/score.cu src/spec.cu)
add_library(fnl_net STATIC ${FNL_SOURCES} src/cpu_experts.cpp)
set_source_files_properties(src/cpu_experts.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=znver4")
target_include_directories(fnl_net PUBLIC src)
target_link_libraries(fnl_net PUBLIC crit::core crit::cuda CUDA::cudart)
target_compile_options(fnl_net PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)

add_executable(fnl_test tools/fnl_test.cu)
target_link_libraries(fnl_test PRIVATE fnl_net)

add_executable(gemm_test tools/gemm_test.cu)
target_link_libraries(gemm_test PRIVATE fnl_net)

add_library(fnl SHARED src/engine.cu)
target_link_libraries(fnl PRIVATE fnl_net)
set_target_properties(fnl PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(gemm_bench tools/gemm_bench.cu)
target_link_libraries(gemm_bench PRIVATE fnl_net)

add_executable(spec_test tools/spec_test.cu)
target_link_libraries(spec_test PRIVATE fnl_net)

add_executable(gemv_bench tools/gemv_bench.cu)
target_link_libraries(gemv_bench PRIVATE fnl_net)

add_executable(miss_bench tools/miss_bench.cu)
target_link_libraries(miss_bench PRIVATE fnl_net)


add_executable(expert_bench tools/expert_bench.cu)
target_link_libraries(expert_bench PRIVATE fnl_net)

add_executable(mx8_bench tools/mx8_bench.cu)
target_link_libraries(mx8_bench PRIVATE fnl_net)
