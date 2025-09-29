cmake_minimum_required(VERSION 3.10)
project(heat_diffusion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable MPI support
option(USE_MPI "Enable MPI support" ON)

# Enable Score-P (optional)
option(USE_SCOREP "Enable Score-P profiling" OFF)

# Use Score-P compiler wrappers if profiling is enabled
if(USE_MPI AND USE_SCOREP)
    set(CMAKE_C_COMPILER scorep-mpicc)
    set(CMAKE_CXX_COMPILER scorep-mpicxx)
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DUSE_MPI)
endif()

add_executable(heat_diffusion heat_diffusion.cpp)

if(USE_MPI)
    target_include_directories(heat_diffusion PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(heat_diffusion MPI::MPI_CXX)
endif()

# Add compiler warnings
if(MSVC)
    target_compile_options(heat_diffusion PRIVATE /W4)
else()
    target_compile_options(heat_diffusion PRIVATE -Wall -Wextra)
endif()
