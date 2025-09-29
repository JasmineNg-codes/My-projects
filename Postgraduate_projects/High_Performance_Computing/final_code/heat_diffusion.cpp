/// \mainpage 2D Heat Diffusion Documentation
///
/// \section opt1_title Final optimisationed/parallelised code
///
/// Welcome to the coursework documentation for the 2D Heat Diffusion simulation.  
/// This repository contains several folders representing different optimization and parallelization stages:  

/// - `baseline_performance`: Reference version  
/// - `optimisation_1_cache`: Optimized for cache usage  
/// - `optimisation_2_blocking`: Implements loop blocking  
/// - `parallel_3_mpi`: Parallelized using MPI  
/// - `parallel_4_openMP`: Parallelized using OpenMP  
/// - `final_code`: Final version with all effective optimizations applied

/// This particular documentation is for \b `final_code`.
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
#include <fstream>
#include <mpi.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <unistd.h>  // for gethostname
#include <omp.h>

// Global grid dimensions and simulation parameters
const int global_width = 100;
const int global_height = 100;
const int steps = 100;
const double diffusionRate = 0.1;

// Function to save the full grid to a file (only on rank 0)
void saveFrame(const std::vector<double>& fullGrid, int frame, int width, int height) {
    std::string filename = "output/frame_" + std::to_string(frame) + ".txt";
    std::ofstream out(filename);
    if (out.is_open()) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                out << fullGrid[y * width + x] << " ";
            }
            out << "\n";
        }
        out.close();
    }
}

int main(int argc, char** argv) {
    int runs = 5;
    if (argc > 1) runs = std::stoi(argv[1]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print hostname for each rank
    char hostname[256];
    gethostname(hostname, 256);
    std::cout << "Rank " << rank << " running on " << hostname << std::endl;

    // Set up a 2D Cartesian topology
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &up, &down);

    // Local grid size (each rank gets a subgrid)
    int local_width = global_width / dims[0];
    int local_height = global_height / dims[1];
    int padded_width = local_width + 2;  // for halo
    int padded_height = local_height + 2;

    // Define a column MPI datatype for vertical halo exchange
    MPI_Datatype column_type;
    MPI_Type_vector(local_height, 1, padded_width, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    for (int run = 0; run < runs; ++run) {
        std::vector<double> temperature(padded_width * padded_height, 20.0);
        std::vector<double> nextTemperature(padded_width * padded_height, 20.0);

        // Hot spot initialization in the center of the global grid
        int global_cx = global_width / 2;
        int global_cy = global_height / 2;
        int local_x_start = coords[0] * local_width;
        int local_y_start = coords[1] * local_height;

        for (int y = 0; y < local_height; ++y) {
            for (int x = 0; x < local_width; ++x) {
                int gx = local_x_start + x;
                int gy = local_y_start + y;
                if (std::abs(gx - global_cx) <= 3 && std::abs(gy - global_cy) <= 3) {
                    temperature[(y + 1) * padded_width + (x + 1)] = 100.0;
                }
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < steps; ++step) {
            // Perform halo exchange with neighbors
            MPI_Sendrecv(&temperature[1 * padded_width + 1], local_width, MPI_DOUBLE,
                         up, 0,
                         &temperature[(local_height + 1) * padded_width + 1], local_width, MPI_DOUBLE,
                         down, 0, cart_comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv(&temperature[local_height * padded_width + 1], local_width, MPI_DOUBLE,
                         down, 1,
                         &temperature[0 * padded_width + 1], local_width, MPI_DOUBLE,
                         up, 1, cart_comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv(&temperature[1 * padded_width + 1], 1, column_type,
                         left, 2,
                         &temperature[1 * padded_width + (local_width + 1)], 1, column_type,
                         right, 2, cart_comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv(&temperature[1 * padded_width + local_width], 1, column_type,
                         right, 3,
                         &temperature[1 * padded_width + 0], 1, column_type,
                         left, 3, cart_comm, MPI_STATUS_IGNORE);

            #pragma omp parallel for schedule(static)

            for (int y = 1; y <= local_height; ++y) {
                int row = y * padded_width;
                for (int x = 1; x <= local_width; ++x) {
                    int idx = row + x;
                    double laplacian =
                        temperature[idx - 1] + temperature[idx + 1] +
                        temperature[idx - padded_width] + temperature[idx + padded_width] -
                        4.0 * temperature[idx];
                    nextTemperature[idx] = temperature[idx] + diffusionRate * laplacian;
                }
            }

            temperature.swap(nextTemperature);

            // Gather results every 10 steps
            if (step % (steps / 10) == 0) {
                std::vector<double> localData(local_width * local_height);
                for (int y = 0; y < local_height; ++y) {
                    for (int x = 0; x < local_width; ++x) {
                        localData[y * local_width + x] =
                            temperature[(y + 1) * padded_width + (x + 1)];
                    }
                }

                std::vector<double> fullGrid;
                if (rank == 0)
                    fullGrid.resize(global_width * global_height);

                MPI_Gather(localData.data(), local_width * local_height, MPI_DOUBLE,
                           fullGrid.data(), local_width * local_height, MPI_DOUBLE,
                           0, cart_comm);

                // Reorder gathered data into the full grid
                if (rank == 0) {
                    std::vector<double> ordered(global_width * global_height);
                    for (int r = 0; r < size; ++r) {
                        int coords_r[2];
                        MPI_Cart_coords(cart_comm, r, 2, coords_r);
                        int x0 = coords_r[0] * local_width;
                        int y0 = coords_r[1] * local_height;

                        for (int j = 0; j < local_height; ++j) {
                            for (int i = 0; i < local_width; ++i) {
                                int global_idx = (y0 + j) * global_width + (x0 + i);
                                int local_idx = r * local_width * local_height + j * local_width + i;
                                ordered[global_idx] = fullGrid[local_idx];
                            }
                        }
                    }
                    saveFrame(ordered, step / 10, global_width, global_height);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        if (rank == 0) {
            std::cout << duration.count() << std::endl;
        }
    }

    MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}
