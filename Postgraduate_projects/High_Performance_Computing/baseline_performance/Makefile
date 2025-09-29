BUILD_DIR = build

all: $(BUILD_DIR)/heat_diffusion

$(BUILD_DIR)/heat_diffusion: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(BUILD_DIR)/heat_diffusion
	mkdir -p output
	./$(BUILD_DIR)/heat_diffusion 20 | tee -a output/time.log
	bash -c 'rm -f output/frame_{10..199}.txt'

valgrind-profile: $(BUILD_DIR)/heat_diffusion
	valgrind --tool=callgrind --simulate-cache=yes ./$(BUILD_DIR)/heat_diffusion 1
	callgrind_annotate $(ls callgrind.out.*) > callgrind_summary.txt

clean:
	rm -rf $(BUILD_DIR)
	rm -rf output
	rm -f slurm-*.out
	rm -f callgrind.out.*
	rm -f callgrind_summary.txt

.PHONY: all clean run
