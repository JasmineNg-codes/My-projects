cmake_minimum_required(VERSION 3.14)
project(HeatDiffusionTest)
set(CMAKE_CXX_STANDARD 17)

# GoogleTest setup
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_heat test.cpp)
target_link_libraries(test_heat gtest_main)
add_test(NAME CompareOutputs COMMAND test_heat)
