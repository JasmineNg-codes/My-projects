/// \mainpage 2D Heat Diffusion Documentation
///
/// \section opt1_title Baseline Performance
///
/// Welcome to the coursework documentation for the 2D Heat Diffusion simulation.  
/// This repository contains several folders representing different optimization and parallelization stages:  
///
/// - `baseline_performance`: Reference version  
/// - `optimisation_1_cache`: Optimized for cache usage  
/// - `optimisation_2_blocking`: Implements loop blocking  
/// - `parallel_3_mpi`: Parallelized using MPI  
/// - `parallel_4_openMP`: Parallelized using OpenMP  
/// - `final_code`: Final version with all effective optimizations applied
///
/// This particular documentation is for \b `baseline_performance`.
///
/// Each folder includes a Doxygen configuration file that generates documentation in `docs/html/index.html`.  
/// The generated documentation in each folder includes:
/// - The base coursework-provided comments  
/// - **Additional explanations** describing the specific optimization or parallelization applied in that version  
///   (highlighted in **bold** above the original documentation)
///
///
/// Documentation is generated using [Doxygen](https://www.doxygen.nl/).

/**
 * @file heat_diffusion.cpp
 * @brief 2D Heat Diffusion Simulation using the Finite Difference Method
 * 
 * This program simulates heat diffusion on a 2D grid using the heat equation:
 * ∂T/∂t = α∇²T
 * where T is temperature, t is time, and α is the thermal diffusivity coefficient.
 * 
 * The simulation uses an explicit finite difference method with a discrete
 * Laplacian operator for spatial derivatives. The boundary conditions are
 * insulating (no heat flux) at the edges.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>

/**
 * @class HeatDiffusion
 * @brief Class implementing the 2D heat diffusion simulation
 * 
 * Uses a double-buffered grid approach to update temperatures and
 * implements the explicit finite difference method for the heat equation.
 */
class HeatDiffusion {
private:
    int width, height;              ///< Dimensions of the simulation grid
    double diffusionRate;           ///< Thermal diffusivity coefficient
    std::vector<std::vector<double>> temperature;    ///< Current temperature grid
    std::vector<std::vector<double>> nextTemperature; ///< Buffer for next timestep

    /**
     * @brief Saves the current temperature grid to a file
     * @param frameNumber The current frame number for the output filename
     * 
     * Creates a space-separated text file containing the temperature values.
     * Files are saved in the 'output' directory with names 'frame_X.txt'
     * where X is the frame number.
     */
    void saveFrame(int frameNumber) {
        system("mkdir -p output");
        std::string filename = "output/frame_" + std::to_string(frameNumber) + ".txt";
        std::ofstream outFile(filename);
        
        if (outFile.is_open()) {
            for (const auto& row : temperature) {
                for (const auto& temp : row) {
                    outFile << temp << " ";
                }
                outFile << "\n";
            }
            outFile.close();
        }
    }

public:
    /**
     * @brief Constructor for the HeatDiffusion simulation
     * @param w Width of the simulation grid
     * @param h Height of the simulation grid
     * @param rate Thermal diffusivity coefficient
     * 
     * Initializes the temperature grids and sets up initial conditions
     * with a hot spot in the center of the grid.
     */

    HeatDiffusion(int w, int h, double rate) 
        : width(w), height(h), diffusionRate(rate) {
        // Initialize grids with ambient temperature (20°C)
        temperature = std::vector<std::vector<double>>(height, std::vector<double>(width, 20.0));
        nextTemperature = temperature;

        // Set up initial conditions - hot spot in the middle (100°C)
        int centerX = width / 2;
        int centerY = height / 2;
        for (int y = centerY - 3; y <= centerY + 3; y++) {
            for (int x = centerX - 3; x <= centerX + 3; x++) {
                if (y >= 0 && y < height && x >= 0 && x < width) {
                    temperature[y][x] = 100.0;
                }
            }
        }
    }

    /**
     * @brief Updates the simulation by one timestep
     * 
     * Implements the finite difference method:
     * T(t+dt) = T(t) + α * [T(x+1,y) + T(x-1,y) + T(x,y+1) + T(x,y-1) - 4T(x,y)]
     * 
     * Uses double buffering to properly update all cells simultaneously.
     * Saves the temperature grid every 10 timesteps.
     */
    void update() {
        static int frameCount = 0;
        
        // Calculate new temperatures using the heat equation
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                // Discrete Laplacian operator
                double laplacian = 
                    temperature[y+1][x] + temperature[y-1][x] + 
                    temperature[y][x+1] + temperature[y][x-1] - 
                    4 * temperature[y][x];

                nextTemperature[y][x] = temperature[y][x] + diffusionRate * laplacian;
            }
        }

        // Swap buffers
        temperature.swap(nextTemperature);
        
        // Save frame every 10 timesteps
        if (frameCount % 10 == 0) {
            saveFrame(frameCount / 10);
        }
        frameCount++;

    }
};

/**
 * @brief Main function running the heat diffusion simulation
 * 
 * Creates a 100x100 grid with diffusion rate 0.1 and runs for 100 timesteps.
 * Output files are saved in the 'output' directory and can be visualized
 * using the accompanying plot_heat.py script.
 */
int main(int argc, char* argv[]) {
    int runs = 5;  // Default number of runs

    // If an argument is provided, use it as the number of runs
    if (argc > 1) {
        runs = std::stoi(argv[1]);
    }
    
    for (int r = 0; r < runs; ++r) {
        HeatDiffusion simulation(100, 100, 0.1);
        auto start = std::chrono::high_resolution_clock::now();

        // Run simulation for 100 frames
        const int TOTAL_FRAMES = 100;
        for (int i = 0; i < TOTAL_FRAMES; i++) {
            simulation.update();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << duration.count() << std::endl;
        std::cout.flush();  // Ensures output is written immediately
    } 
    return 0;
}