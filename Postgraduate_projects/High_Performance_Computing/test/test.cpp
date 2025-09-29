#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>

//First, I want to define a function called load grid.
//This function will take the file name and then load the frame from the file into my vector in vector grid.
std::vector<std::vector<double>> load_grid(const std::string& filename) {
    std::ifstream file(filename);
    //initialise input file stream, and provide a print statement if it failed to open
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::vector<double>> grid; //define the gird
    std::string line; //define a line of text

    while (std::getline(file, line)) { //read the file line by line
        std::istringstream iss(line); //fill a new input string stream (called iss) with the line
        std::vector<double> row;
        double val;
        while (iss >> val) row.push_back(val); //put the stream input into val variable, and then keep adding into the row.
        if (!row.empty()) grid.push_back(row); //if the current row not empty, then add the completed row to grid.
    }

    return grid; //now this function will have copied over a full grid values into my new grid in test.cpp.
}

//MSE function
double compute_mse(const std::vector<std::vector<double>>& a,
                   const std::vector<std::vector<double>>& b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        std::cerr << "Grid size mismatch\n"; //CHECK DIEMNSIONS ARE THE SAME
        std::exit(EXIT_FAILURE);
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) //loop through rows
        for (size_t j = 0; j < a[0].size(); ++j) //loop through columns
            sum += std::pow(a[i][j] - b[i][j], 2); //for that one elemnet, calculate ms3

    return sum / (a.size() * a[0].size());
}

// Folders to compare
const std::string baseline_folder = "../../baseline_performance/output";
const std::vector<std::string> optim_folders = {
    "../../optimisation_1_cache/output",
    "../../optimisation_2_blocking/best_block/output",
    "../../parallel_3_mpi/output",
    "../../parallel_4_openMP/output",
    "../../final_code/output"
};

//test
TEST(HeatDiffusionComparison, CompareAllFrames) {
    for (const auto& folder : optim_folders) {
        bool all_frames_passed = true;

        for (int frame = 0; frame < 10; ++frame) {
            std::string filename = "frame_" + std::to_string(frame) + ".txt";
            std::string baseline_path = baseline_folder + "/" + filename;
            std::string test_path = folder + "/" + filename;

            std::cout << "Comparing " << test_path << " to baseline...\n";

            auto baseline = load_grid(baseline_path);
            auto candidate = load_grid(test_path);
            double mse = compute_mse(baseline, candidate);

            std::cout << "MSE = " << mse << std::endl;

            if (mse >= 1e-6) {
                std::cerr << "High MSE for " << filename << " in " << folder << "\n";
                all_frames_passed = false;
            }
            //This is the google test of comparing whether MSE is smaller than 1e-6

            EXPECT_LT(mse, 1e-6) << "High MSE for " << filename << " in " << folder;
        }

        if (all_frames_passed) {
            std::cout << "The folder: " << folder << " passed all tests!\n";
        }
    }
}

