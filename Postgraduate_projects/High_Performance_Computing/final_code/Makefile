# MPI build directory
BUILD_DIR = build

# Number of MPI Processes (Default: 4)
PROCS ?= 8

# Default target: Build MPI version
mpi: $(BUILD_DIR)/heat_diffusion

$(BUILD_DIR)/heat_diffusion: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DUSE_MPI=ON .. && make

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Run the MPI heat diffusion simulation
mpi_run: $(BUILD_DIR)/heat_diffusion
	mkdir -p output
	mpirun -np $(PROCS) ./build/heat_diffusion 20 | tee -a output/time.log

# Clean MPI build
mpi_clean:
	rm -rf $(BUILD_DIR)
	rm -rf output
	rm -f slurm-*.out
	rm -f callgrind.out.*
	rm -f callgrind_summary.txt

.PHONY: mpi mpi_run mpi_clean valgrind-profile