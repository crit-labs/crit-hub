cmake_minimum_required(VERSION 3.28)
project(crit_core_forge_projection LANGUAGES CXX)

# Build the runtime from this checkout; no installed SDK is required.
set(BUILD_TESTING OFF CACHE BOOL "Build the full Core fixture suite")
get_filename_component(CRIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_subdirectory("${CRIT_ROOT}/core" "${CMAKE_CURRENT_BINARY_DIR}/core")

add_executable(crit_core_forge_projection main.cpp)
target_link_libraries(crit_core_forge_projection PRIVATE crit::core)
target_include_directories(crit_core_forge_projection PRIVATE
  "${CRIT_ROOT}/forge/quantized-projections/code")
crit_strict_warnings(crit_core_forge_projection)

enable_testing()
add_test(NAME core_forge_projection COMMAND crit_core_forge_projection)
set_tests_properties(core_forge_projection PROPERTIES TIMEOUT 30)
