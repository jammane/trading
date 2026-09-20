cmake_minimum_required(VERSION 3.16)
project(trading_cpp CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -ffast-math implies -ffinite-math-only, under which GCC folds std::isfinite()/isnan() to
# constants. That silently disabled EVERY finiteness guard in the trainer: measured, the
# order-intent sell aggregate came back NaN on 98.55% of industry-days because its only guard
# was an isfinite() that never ran, and `cn_ok()` -- MT1CNet's input sanitiser -- was a no-op.
# Comparisons (`x > 0.f`) still reject NaN, which is why the buy-side twin survived and the
# failure looked like a one-off rather than a class. Keep the rest of fast-math; turn this one
# sub-flag back on. tests/test_fp_guards.cpp pins it and is built with the SAME flags.
add_compile_options(-O3 -march=native -ffast-math -fno-finite-math-only)

find_package(BLAS REQUIRED)

# Locate cblas.h — may be in a subdirectory like openblas/
find_path(CBLAS_INCLUDE_DIR cblas.h
  HINTS /usr/include/openblas /usr/local/include/openblas
        /usr/include /usr/local/include)

add_executable(training_v4_cpp training_v4.cpp)
# The binary that matters carried NO warning flags until v0.8.1.8 -- only the test targets did.
# That is why two positional MasterResult aggregates sat here unflagged, the same shape as the
# IndResult bug that shipped. Warnings on the trainer are the thing that makes the class visible.
target_compile_options(training_v4_cpp PRIVATE -Wall -Wextra)
target_include_directories(training_v4_cpp PRIVATE ${CBLAS_INCLUDE_DIR})
target_link_libraries(training_v4_cpp ${BLAS_LIBRARIES} pthread)

enable_testing()

# Pass-boundary seeding unit tests — also pure math, no BLAS
add_executable(test_pass_seeding_cpp tests/test_pass_seeding.cpp)
target_include_directories(test_pass_seeding_cpp PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(test_pass_seeding_cpp PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_pass_seeding_cpp m)
add_test(NAME test_pass_seeding COMMAND test_pass_seeding_cpp)

# Floating-point guards must survive the optimiser. Built with the DIRECTORY's flags on
# purpose -- no -O override that could mask a fast-math regression -- so dropping
# -fno-finite-math-only breaks the build instead of silently voiding every isfinite() check.
add_executable(test_fp_guards_cpp tests/test_fp_guards.cpp)
target_include_directories(test_fp_guards_cpp PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(test_fp_guards_cpp PRIVATE -Wall -Wextra)
target_link_libraries(test_fp_guards_cpp m)
add_test(NAME test_fp_guards COMMAND test_fp_guards_cpp)

# Single-output MT1 scoring + pool lifecycle (the rebuild) — pure math, no BLAS
add_executable(test_mt1_pool_cpp tests/test_mt1_pool.cpp)
target_include_directories(test_mt1_pool_cpp PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(test_mt1_pool_cpp PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_mt1_pool_cpp m)
add_test(NAME test_mt1_pool COMMAND test_mt1_pool_cpp)
