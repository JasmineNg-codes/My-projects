cmake_minimum_required(VERSION 3.10)
project(heat_diffusion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add executable
add_executable(heat_diffusion heat_diffusion.cpp)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(heat_diffusion PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add compiler warnings
if(MSVC)
    target_compile_options(heat_diffusion PRIVATE /W4)
else()
    target_compile_options(heat_diffusion PRIVATE -Wall -Wextra)
endif()

# Optional: print OpenMP info
message(STATUS "OpenMP Flags: ${OpenMP_CXX_FLAGS}")
