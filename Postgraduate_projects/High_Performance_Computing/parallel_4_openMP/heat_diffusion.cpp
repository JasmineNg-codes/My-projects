/// \mainpage 2D Heat Diffusion Documentation
///
/// \section opt1_title Parallelisation 5: OpenMP
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
/// This particular documentation is for \b `parallel_5_openMP`.
///
/// Each folder includes a Doxygen configuration file that generates documentation in `docs/html/index.html`.  
/// The generated documentation in each folder includes:
/// - The base coursework-provided comments  
/// - **Additional explanations** describing the specific optimization or parallelization applied in that version  
///   (highlighted in **bold** above the original documentation)
///
///
/// Documentation is generated using [Doxygen](https://www.doxygen.nl/).

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <omp.h>

/**
 * @class HeatDiffusion
 * @brief Class implementing the 2D heat diffusion simulation
 * @ingroup openmp_module
 * This class models heat diffusion across a 2D grid using the explicit finite difference method.
 * 
 * **Optimisation implemented**: **flattened 1D grid** for cache-friendly access and applies OpenMP
 * to parallelize the core update step. 
 */
class HeatDiffusion {
private:
    int width, height;
    double diffusionRate;
    std::vector<double> temperature;
    std::vector<double> nextTemperature;

    void saveFrame(int frameNumber) {
        system("mkdir -p output");
        std::string filename = "output/frame_" + std::to_string(frameNumber) + ".txt";
        std::ofstream outFile(filename);

        if (outFile.is_open()) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    outFile << temperature[y * width + x] << " ";
                }
                outFile << "\n";
            }
            outFile.close();
        }
    }

/**
     * @brief Constructor for the HeatDiffusion simulation
     * @param w Width of the simulation grid
     * @param h Height of the simulation grid
     * @param rate Thermal diffusivity coefficient
     * 
     * Initializes the temperature grids and sets up initial conditions
     * with a hot spot in the center of the grid.
     */
public:
    HeatDiffusion(int w, int h, double rate)
        : width(w), height(h), diffusionRate(rate),
          temperature(w * h, 20.0),
          nextTemperature(w * h, 20.0) {

        int centerX = width / 2;
        int centerY = height / 2;

        for (int y = centerY - 3; y <= centerY + 3; y++) {
            for (int x = centerX - 3; x <= centerX + 3; x++) {
                if (y >= 0 && y < height && x >= 0 && x < width) {
                    temperature[y * width + x] = 100.0;
                }
            }
        }
    }

    /**
     * @brief Updates the temperature grid by one timestep (parallelized with OpenMP)
     * 
     * Implements the finite difference method:
     * T(t+dt) = T(t) + α * [T(x+1,y) + T(x-1,y) + T(x,y+1) + T(x,y-1) - 4T(x,y)]
     * 
     * This function is parallelized with OpenMP using `#pragma omp parallel for`.
     * 
     * **Optimizations implemented**:
     * - **OpenMP parallelization** across rows for improved performance on multicore systems
     * - **Flattened 2D grid** to improve memory access locality
     */
    void update() {
        static int frameCount = 0;
        #pragma omp parallel for schedule(static)
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int index = y * width + x;  // 

                double top    = temperature[index - width];
                double bottom = temperature[index + width];
                double left   = temperature[index - 1];
                double right  = temperature[index + 1];
                double center = temperature[index];

                double laplacian = top + bottom + left + right - 4 * center;
                nextTemperature[index] = center + diffusionRate * laplacian;
            }
        }

        temperature.swap(nextTemperature);

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
 * **Optimisation implemented**: **Sets the number of OpenMP threads (default is max available)** and runs the heat diffusion
 * **The number of threads can be overridden via command-line argument**.
 */
int main(int argc, char* argv[]) {
    int runs = 5;
    //I set the default to max threads in the system
    int threads = omp_get_max_threads(); 

    if (argc > 1) {
        runs = std::stoi(argv[1]);
    }

    if (argc > 2) {
        threads = std::stoi(argv[2]);
    }
    //But if i set the threads then it will override the previous comments
    omp_set_num_threads(threads);

    //Print out the number of threads I will be running in the time log
    std::cout << "Running with " << threads << " thread(s)" << std::endl;

    for (int r = 0; r < runs; ++r) {
        HeatDiffusion simulation(100, 100, 0.1);
        auto start = std::chrono::high_resolution_clock::now();

        const int TOTAL_FRAMES = 100;
        for (int i = 0; i < TOTAL_FRAMES; i++) {
            simulation.update();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << duration.count() << std::endl;
        std::cout.flush();
    }
    return 0;
}
