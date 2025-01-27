# Project Overview
Swing_Test is the implementation, debug and benchmarking project for the Swing algorithm.

It is a modular project featuring a static library (`libswing.a`), test and benchmark executables (`bin/`), and SLURM-based scripts for benchmarking and debugging on clusters. Future updates will include data analysis and graphing tools.

The suggested workflow is to use and modify the .sh files in `scripts/` directory to set up environments, run debug tests and benchmarking tests.

## Directory Structure

```
.
├── bin                   # Binaries (executables)
├── include               # Header file for libswing library containing the functions' signatures
├── lib                   # Compiled static library
├── libswing              # Libswing library source code
├── makefile              # Top-level Makefile
├── obj                   # Object files
├── plot                  # Python scripts for data analysis and visualization (in development)
├── results               # Results folder
├── test                  # Test program source code, includes benchmarking and debugging
├── ompi_rules            # Open MPI dynamic rule file generator
└── scripts               # Debug and benchmarking scripts
```

## Components

### `libswing` - Static Library

The `libswing/` directory contains the source code for the `libswing/` static library. This library is compiled into a static archive (`lib/libswing.a`) and is used by the other components in the project. It provides essential utilities and functions that are benchmarked and debugged in the subsequent components.

It contains Swing implementations built **OVER** MPI (beware that the tests are thought to work also on internal algorithms of Open MPI).

#### Modify `libswing`
To add new collectives, declare them on `include/libswing.h` where they should be thoroughly documented with a Doxygen style.

Actual implementation must be written in `libswing/libswing_<coll_type>.c` with helper functions declared as `static inline` in `libswing/libswing_utils.h`.

### `test/` - Benchmark program

The `test/` directory contains a set of benchmark tests that are used to measure the performance of the `libswing` library. It compiles the executable `bin/test` that benchmarks the algorithm provided by `libswing` and the ones written inside the `Open MPI` library.

It will be modified to be independent of MPI implementation (i.e. to work also with standard Open MPI and MPICH).

Algorithm selection is done via a command line parameter and, in case of `OMPI_TEST` implementations, environmental variables and a rule file must be update accordingly before running the tests.

The executable itself must be run with `srun` or `mpirun`/`mpiexec` and output is saved in `csv` format by rank 0.

##### Parameters:
- `<array_size>`: Size of the array to run the collective on.
- `<iter>`: Number of total iterations to run of the specific test.
- `<type_string>`: The string codifying the datatype of the test. Currently some of them don't work and documentation will be added.
- `<algorithm>`: Algorithm ID to run tests on.
- `<dirpath>`: Directory where results will be saved.

Collective type instead is chosen via environmental variable `COLLECTIVE_TYPE` (for now `ALLREDUCE`, `ALLGATHER` and `REDUCESCATTER` are implemented, other collectives coming soon).

##### Saving benchmarking results
Before saving the results, a ground truth check on the last iteration is performed to check for possible errors.

For now results are saved as a matrix in which each row is one iteration of the test, the first column contain the highest time between the ranks and each other row is the time of a specific rank.

Naming of this file is temporary and will be changed.

The first time this main is called (i.e. if used with a script test/debug suite, only on the first call), also a .csv file with MPI rank allocations will be saved.

##### Debugging
When compiled with -DDEBUG it will not save benchmark results and, if ground truth check returns an error, it will print `rbuf` and `rbuf_gt` before invoking `MPI_Abort`.

Also in this mode, `sbuf` will be initialized with a predefined sequence of power of 10, in order to help for debugging purpose.

### `ompi_rules/` - Open MPI Rule File Generator

The `ompi_rules/` directory contains the source for a program that generates dynamic rule file for `Open MPI`. These rule files define the collective communication rules that Open MPI uses. The program compiles into the executable `bin/change_dynamic_rules`, which is used to select specified algorithms.

Beware that the following environmental variables must be set after the rule file is generated:
```bash
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
export OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}
```
In the normal workflow, those variables are set up with the scripts to run tests and debug.

It will be modified to work also for `MPICH` algorithm selection.

### `scripts` - Running benchmarks and debug tests
The `scripts/` contains scripts to benchmark and debug the library. It also contains scripts to set up the environment based on where the library is being run.

## Running Tests and Benchmark

The project includes scripts to run the benchmark/debug suite inside the `scripts/` directory.

The suggested workflow is to `$ sbatch` the `scripts/submit.sbatch` script when benchmarking if on cluster, instead run the test\debug suite script on its own if running on local machine.

For now specific variables must be modified inside the run suites but it's planned to bring them on the submit script. Beware that everything for now it's extremely hand tailored on MY workflow and only now I'm polishing the program to let other people work with those scripts so a lot of modifications are still needed.


### `scripts/submit.sbatch` - Submit Tests via SLURM

For cluster-based environments that use SLURM for job scheduling, a template script `scripts/submit.sbatch` is provided. This script can be used to submit the test runs to the cluster via SLURM.

To use it, you will need to customize the script as per your SLURM job submission requirements. The script includes placeholders and options for specifying the number of processes, job time, and output location.

#### Example Usage for `submit.sbatch`

```bash
$ sbatch scripts/submit.sbatch
```
##### Parameters:
- **`<p_name>`:** Name of the partition to run tests on on the target cluster.
- **`<qos_name>`:** Name of the required quality of service. (OPTIONAL)
- **`<n_nodes>`:** The number of nodes to request for the SLURM job. Note that, for benchmarking reasons this must be the number of processes (i.e. one single process for node, irrespective of node cores).
- **`<requested_time>`:** The time requested for job allocation. Higher number of hours is suggested.
- **`<account_name>`:** Account name on the target cluster.
- **`<LOCATION>`:** Name of the machine, as defined in `scripts/environments/`.
- **`<DEBUG_MODE>`:** If run on debug mode.
- **`<TIMESTAMP>`:** Current time-stamp used to create the results folder. Can be changed to anything.
- **`<COLLECTIVE_TO_TEST>`:** Collective type to test.
- **`<ENABLE_CUDA>`:** If use CUDA-aware MPI. Beware that this option works only on Open MPI for now.
- **`<ENABLE_OMPI_TEST>`:** If use custom Open MPI library with swing implementations.

Also the standard output and error will be redirected into the results directory.

It will be modified to allow for algorithm selection directly in this stage. 

###### Warning
Beware that, especially with big allocations, those scripts can fail and waste compute hours if left uncheck. It's suggested, if possible, for big allocations, to check if the script starts working. If it does it's unlikely that compute time will be wasted. Currently working on the reliability of this part.

### `scripts/run_test_suite.sh` - Benchmarking Suite
This script runs a benchmarking test suite itself.

To run it without relying to the `submit.sbatch` script execute the following command:

```bash
$ scripts/run_test_suite.sh <N_NODES> [COLLECTIVE_TYPE] [DEBUG_MODE] [TIMESTAMP] [LOCATION] [ENABLE_CUDA] [ENABLE_OMPI_TEST]
```

##### Parameters:
- **`<N_NODES>`** (required): The number of processes to use for the test run. Must be a valid positive integer greater than or equal to 2.
- **`[COLLECTIVE_TYPE]`** (optional): Which collective to test. Defaults to `ALLREDUCE`
- **`[DEBUG_MODE]`** (optional): Flag to enable debug mode . Defaults to `no`.
- **`[TIMESTAMP]`** (optional): Timestamp for the test run. Defaults to the current date and time. It works as the output (sub)directory of the test and can be set to anything.
- **`[LOCATION]`** (optional): Specifies the environment configuration script. Defaults to `local`.
- **`[ENABLE_CUDA]`** (optional): Enable CUDA aware MPI support. Defaults to `no`.
- **`[ENABLE_OMPI_TEST]`** (optional): Enable the use of the modified Open MPI library with Swing Allreduce algorithms. Defaults to `yes`.

##### Key Features:
- Validates and initializes input parameters.
- Dynamically sets up environment and configurations.
- Compiles the codebase.
- Runs benchmarks for specified collective algorithms.

Test results are saved in the directory structure `$SWING_DIR/results/<location>/<timestamp>/data/`, alongside with the given node allocation in `$SWING_DIR/results/<location>/<timestamp>/alloc.csv`.

Moreover, the script also invokes an updater of the `$SWING_DIR/results/<location>/<location>_metadata.csv` file, adding a row with the metadata of the last test.
THIS FEATURE IS STILL IN DEVELOPMENT, IT HAS BEEN ADDED TO STREAMLINE LIBRARY TESTING.

**NOTE:** The script cleans and rebuilds the codebase since if `ENABLE_OMPI_TEST` or `DEBUG` are set to `yes`, the `test` and `change_dynamic_rules` binaries must be injected with `-DOMPI_TEST` or `-DDEBUG` (debug is specific only to test).

##### Key Variables:
- **COLLECTIVE_ALGOS**: Lists supported algorithms for each collective type.
- **COLLECTIVE_SKIPS**: Specifies algorithms to skip for certain configurations.
- **ARR_SIZES**: Array sizes used for benchmarking.
- **TYPES**: Data types for testing (default: `int64`).


#### Utils Script
Provides utility functions to support the main script such as:
1. **`error`**: Prints error messages in red.
2. **`success`**: Prints success messages in green.
3. **`cleanup`**: Handles SIGINT (Ctrl+C) signals and kills all child processes.
4. **`source_environment`**: Sources the environment configuration based on the location.
5. **`compile_code`**: Cleans and compiles the codebase with optional flags for OpenMPI tests and debug mode.
6. **`get_iterations`**: Determines the number of iterations for benchmarking based on array size.
7. **`run_test`**: Executes a single test case with specified parameters.
8. **`run_all_tests`**: Iterates through configurations and runs tests for each combination.

#### Add new environments
The script automatically sets up required environment variables based on the `location` parameter. To define a new environment configuration:
1. Create a new script in `scripts/environments/`.
2. Export the necessary variables.
3. Update the `location` parameter to point to the new script.

#### Debug mode
When running on debug mode, manually select array sizes, algorithms and datatypes to run tests on. Debug mode will not save benchmark test time or allocations.

## Data Analysis and Visualization

Still in development. For now only two small scripts are added since I'm preparing everything to be used by more people.

Take a look, but it's still WIP.


## Building

To build the project, use the following steps:

1. **Build all components**:
   ```bash
   $ make
   ```

2. **Clean all builds**:
   ```bash
   $ make clean
   ```

The `Makefile` automatically handles the compilation and linking for each component, creating the necessary binaries and libraries in the `bin` and `lib` directories.

### Compiling Individual Parts of the Project
The project is built with a recursive makefile approach in which a top-level `Makefile` will run the `make` commands for each subfolder.

So, to build everything just run the `make` command in the main directory. You can also use the `make` command with the `clean` target to remove object files and binaries.

If you want to compile and build individual parts of the project you can either run the `make` command inside the desired subdirectory, or run `make -C <directory>`. This works also with the `clean` command.

## Results Management

In this repository, the results are stored in a directory that is automatically compressed into a `.tar` file to save storage space and keep the repository clean. The process works as follows:

1. **Results directory:** All raw results are saved in a dedicated directory, different for each system on which tests are run.
2. **Test metadata:** For each new test, `scripts/run_test_suite.sh` will invoke `results/metadata.py` script, to add the metadata of the new test into a `.csv` file containing metadata of tests ran on a given system.
3. **Compression Script:** A script is included in the repository to compress the results directory into a `.tar` file. The script also add to `.gitignore` the uncompressed results subdirectory and remove possible duplicates from the `.gitignore` file.
4. **Pre-Commit Hook:** A pre-commit hook is set up to run the compression script automatically before each commit. This ensures that any new results are compressed and added to the repository in a consistent manner.

There is no need to run the compression or the metadata script manually as everything is done automatically by the test suite script or the pre-commit hook.


## TODO
#### Makefile
- [x] add error handling when building\linking the library (done by the test script)
- [ ] make the current static `libswing.a` a dynamic `libswing.so` to be added with `LD_PRELOAD`
#### Libswing modifications
- [x] prepare for the possibility of implementing different collectives by refactoring code
- [ ] debug allgather swing static
- [ ] write reduce scatter swing
- [ ] document functions and comment code
#### Test program
- [x] document functions and comment code
- [x] create a function to write rank and allocation inside test without relying on normal `srun` in test suite
- [x] use enum when possible for clarity
- [x] standardize .csv format to what was decided with professor De Sensi
- [x] create a general interface to select specific testing for specific collectives without duplicating code by adding a sea of if-else statements (in particular modify the test loop to use a function pointer for each specific allreduce function and a switch for other collectives)
- [x] separate and modularize ground truth check for different kinds of collectives (WIP)
- [x] finish logic for allgather
- [x] finish logic for reducescatter
- [ ] implement logic for allgather_k_bruck (radix selection) and debug both internal and external
#### OMPI rules
- [x] allow for multiple types of algorithm selection
- [ ] change it so that it recognizes if the modified `Open MPI` is being run or not
- [ ] modify to let it work also with `MPICH` algorithm selection
#### Test suite
- [x] add error handling and more explicit messages about what is being done
- [x] separate results by system
- [x] comment the code
- [x] create a function to add metadata in the .csv asked by the professor
- [x] build a better and clearer interface to select variables for testing
- [x] integrate different collective testing
- [ ] ensure python modules and environment is set up correctly
- [ ] add an env var to select between `MPICH` and `Open MPI` binaries, independently of `ompi_test` (obviously `ompi_test` must be no when MPICH is selected)
#### Submit Script
- [x] automatically inject `$N_NODES` inside suite based on selected `-N` without further modifications
- [x] insert an `<output_directory>` for stderr and stdout of slurm
- [ ] bring variables to select inside test/debug suites to this layer in order to give a better interface. Modifications on those suite is needed when running test without the submit script
#### Results folder
- [x] add a script to compress the data
- [x] add pre-commit hook that triggers the compression of the new data, adds the uncompressed data to the gitignore
- [x] add the script to build and update the .csv description
- [ ] add tests present on my systems (or maybe not)
#### Plot python scripts
- [x] add the possibility of selecting specific tests
- [x] document and comment code
- [ ] adjust everything to new structure
- [ ] avoid working only with summaries, or at least give the possibility of not doing it
- [ ] improve naming of output graphs so that they don't get overwritten by similar graphs
- [ ] refactor to separate graph drawing functions to separate files
