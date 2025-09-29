# Heat Diffusion Optimization and Parallelization Repository

This repository showcases the development steps for optimizing a heat diffusion code, aiming to achieve a 2x speed improvement from the baseline. Various optimization techniques like flattening and cache blocking were implemented and tuned, while parallelization approaches including MPI and OpenMP were explored through 2D Cartesian decomposition and experimentation with different ranks (MPI) and thread numbers (OpenMP). Each folder contains the optimised or parallelised code that yielded the best performance after tuning.

## Features

This repository conducts automatic unit testing using Google Test framework and Mean Squared Error (MSE) validation to ensure outputs from new optimizations match baseline results. To contribute a new optimization, create a new branch, add a dedicated folder for your optimization, and update the test.cpp file to include your folder directory. Tests can be run on both local machines and the CSD3 supercomputer.

## Usage

### Executing program

#### On csd3

To run all job scripts simultaneously on CSD3, use:
```
sbatch all_scripts
```
Alternatively, cd into the desired folder and run:
```
sbatch job.sh
```
#### On local machine

The optimizations can be run on a local laptop by navigating to your desired optimization folder and executing:
```
make clean
make
make run
```
The execution times will be displayed in your terminal and logged in time.log within the output folder. 

## Running tests

### With branching

Automatic unit testing on all the optimised/parallelised code could be done through git branching:
```
git checkout -b branch_name
git add .
git commit -m "testing all outputs"
git push -u origin branch_name
```

Then navigate to the link for the merge request, where automated tests will run. An administrator will review and accept your merge if tests pass.

### In test folder

Alternatively, you can test the folders locally by navigating into the test folder and running the following commands:

```
make clean
make
```
This should tell you whether you have passed the test in your terminal, and if an error has occured, which folder and which frame caused the error. 

## Changing the parameters

1. Number of runs: To modify the number of runs, edit the Makefile to change `./$(BUILD_DIR)/heat_diffusion NUMBER_OF_RUNS`, or navigate to the build directory and run `heat_diffusion NUMBER_OF_RUNS` directly.

2. Number of cache blocks: Change the number of cache blocks in the heat_diffusion.cpp file.

3. Number of processes: To adjust the number of processes used, edit the job.sh script in the optimization_4_mpi_tuning folder by changing `PROCS=desired_number` and updating the tasks parameter in the job script header to 4. 

4. Number of threads: Thread count can be modified by changing the number of tasks. 

5. Number of nodes: To use 2 nodes instead of 1, update the node count at the beginning of the script.

## Documentation

All code is documented using Doxygen, and the generated documentation is located in the docs folder within each corresponding directory. To view the documentation, navigate to the desired folder and run:

```
open docs/html/index.html

```
This will open the main page, which outlines the specific optimisation technique implemented in that folder and the structure of the repository. A class list with documented components, where key optimisations and parallelisation strategies are highlighted in bold.

## Contributing

Contributions via pull requests are welcome!

## License

MIT license. See LICENSE.txt for full license details.