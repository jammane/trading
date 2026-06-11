cmake_minimum_required(VERSION 3.16)
project(trading_cpp CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -march=native -ffast-math)

find_package(BLAS REQUIRED)

add_executable(training_v4_cpp training_v4.cpp)
target_link_libraries(training_v4_cpp ${BLAS_LIBRARIES} pthread)
