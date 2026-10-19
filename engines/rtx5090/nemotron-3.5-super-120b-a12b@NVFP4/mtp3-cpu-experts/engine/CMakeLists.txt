cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES 120a)
project(nmh LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(critCore 0.2.1 EXACT CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(NMH_SOURCES src/gemv.cu src/gemm.cu src/experts.cu src/mamba.cu src/layers.cu src/prefill.cu
    src/fp8_kv_attention.cu src/spec.cu src/lt.cu src/net.cu src/net_decode.cu src/fp4_prompt.cu src/fp4_grouped.cu)
add_library(nmh_net STATIC ${NMH_SOURCES} src/cpu_experts.cpp)
set_source_files_properties(src/cpu_experts.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=znver4;-ffp-contract=off")
# CUTLASS v4.7.1 headers for the grouped NVFP4 prompt GEMMs only (crit kernels dependency third_party/kernels/cutlass).
set(NMH_CUTLASS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kernels/cutlass/artifacts)
set_source_files_properties(src/fp4_grouped.cu PROPERTIES INCLUDE_DIRECTORIES
  "${NMH_CUTLASS}/include;${NMH_CUTLASS}/tools/util/include")
target_include_directories(nmh_net PUBLIC src)
target_link_libraries(nmh_net PUBLIC crit::core crit::cuda CUDA::cudart CUDA::cublasLt)
target_compile_options(nmh_net PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr -O3>)

add_library(nmh SHARED src/engine.cu)
target_link_libraries(nmh PRIVATE nmh_net)
target_compile_options(nmh PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(nmh PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
