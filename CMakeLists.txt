cmake_minimum_required(VERSION 3.16)
project(trading_cpp CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -march=native -ffast-math)

find_package(BLAS REQUIRED)

# Locate cblas.h — may be in a subdirectory like openblas/
find_path(CBLAS_INCLUDE_DIR cblas.h
  HINTS /usr/include/openblas /usr/local/include/openblas
        /usr/include /usr/local/include)

add_executable(training_v4_cpp training_v4.cpp)
target_include_directories(training_v4_cpp PRIVATE ${CBLAS_INCLUDE_DIR})
target_link_libraries(training_v4_cpp ${BLAS_LIBRARIES} pthread)

# MT1 pure-math unit tests — no BLAS dependency
enable_testing()
add_executable(test_mt1_cpp tests/test_mt1.cpp)
target_compile_options(test_mt1_cpp PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_mt1_cpp m)
add_test(NAME test_mt1 COMMAND test_mt1_cpp)
