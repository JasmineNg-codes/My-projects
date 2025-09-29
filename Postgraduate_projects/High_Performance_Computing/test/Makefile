.PHONY: all build run clean

# Default target
all: build run

build:
	mkdir -p build
	cd build && cmake ..

run:
	cd build && make && ctest --output-on-failure -V


clean:
	rm -rf build
