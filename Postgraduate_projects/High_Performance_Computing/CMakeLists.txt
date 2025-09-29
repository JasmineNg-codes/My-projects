cmake_minimum_required(VERSION 3.10)
project(HeatDiffusionProject)

# Add each optimisation folder
#add_subdirectory(optimisation_1_cache)
add_subdirectory(optimisation_2_blocking/best_block)
#add_subdirectory(optimisation_3_combined/best_block)

# Enable testing and include the test suite
enable_testing()
add_subdirectory(test)
