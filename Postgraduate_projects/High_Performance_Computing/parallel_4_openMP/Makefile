# Build directory
BUILD_DIR = build

# Default target: Build OpenMP version
all: $(BUILD_DIR)/heat_diffusion

# Build target
$(BUILD_DIR)/heat_diffusion: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Run the heat diffusion simulation (threads set by environment)
run: $(BUILD_DIR)/heat_diffusion
	mkdir -p output
	./$(BUILD_DIR)/heat_diffusion 20 | tee -a output/time.log
	bash -c 'rm -f output/frame_{10..199}.txt'

# Valgrind profile (single thread)
valgrind-profile: $(BUILD_DIR)/heat_diffusion
	valgrind --tool=callgrind --simulate-cache=yes ./$(BUILD_DIR)/heat_diffusion 1
	callgrind_annotate $(shell ls callgrind.out.* | head -n 1) > callgrind_summary.txt

# Clean build
clean:
	rm -rf $(BUILD_DIR)
	rm -rf output
	rm -f slurm-*.out callgrind.out.* callgrind_summary.txt

.PHONY: all run clean valgrind-profile
