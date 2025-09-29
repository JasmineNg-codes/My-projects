cmake_minimum_required(VERSION 3.10)
project(heat_diffusion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(heat_diffusion heat_diffusion.cpp)

# Add compiler warnings
if(MSVC)
    target_compile_options(heat_diffusion PRIVATE /W4)
else()
    target_compile_options(heat_diffusion PRIVATE -Wall -Wextra)
endif() 