# Project Overview
Swing Allreduce is a modular project featuring a static library (`libswing.a`), test and benchmark executables (`bin/`), and SLURM-based scripts for benchmarking and debugging on clusters. Future updates will include data analysis and graphing tools.

The suggested workflow is to use and modify the .sh files in `scripts/` directory to set up environments, run debug tests and benchmarking tests.

## Directory Structure

```
.
├── bin                   # Binaries (executables)
├── debug                 # Debug program source code
├── include               # Header file for libswing library containing the functions' signatures
├── lib                   # Compiled static library
├── libswing              # Libswing library source code
├── makefile              # Top-level Makefile
├── obj                   # Object files
├── plot                  # Python scripts for data analysis and visualization (in development)
├── results               # Results folder
├── test                  # Test (benchmark) program source code
├── ompi_rules            # Open MPI rule file generator
└── scripts               # Debug and benchmarking scripts
```

## Components

### `libswing` - Static Library

The `libswing/` directory contains the source code for the `libswing/` static library. This library is compiled into a static archive (`lib/libswing.a`) and is used by the other components in the project. It provides essential utilities and functions that are benchmarked and debugged in the subsequent components.

It contains Swing implementations built **OVER** MPI (beware that the tests are thought to work also on internal algorithms of Open MPI).

#### Modify `libswing`
To add new collectives, declare them on `include/libswing.h` where they should be thoroughly documented with a Doxygen style.

Actual implementation must be written in `libswing/libswing.c` with helper functions declared as `static inline` in `libswing/libswing_utils.h`.

Structure will be modified when other collectives will be added.


### `test/` - Benchmark program

The `test/` directory contains a set of benchmark tests that are used to measure the performance of the `libswing` library. It compiles the executable `bin/test` that benchmarks the algorithm provided by `libswing` and the ones written inside the `Open MPI` library.

It will be modified to be independent of MPI implementation (i.e. to work also with standard Open MPI and MPICH).

Algorithm selection is done via a command line parameter and, in case of `Open MPI` implementations, environmental variables and a rule file must be update accordingly before running the tests.

The executable itself must be run with `srun` or `mpirun`/`mpiexec` and output is saved in `csv` format by rank 0.

Parameter required are:
- `<array_size>`: Size of the array to run the collective on.
- `<iter>`: Number of total iterations to run of the specific test.
- `<type_string>`: The string codifying the datatype of the test. Currently some of them don't work and documentation will be added.
- `<alg_number>`: Number of Allreduce algorithm to run tests on
- `<dirpath>`: Directory where results will be saved.

Before saving the results, a ground truth check on the last iteration is performed to check for possible errors.

For now results are saved as a matrix in which each row is one iteration of the test, the first column contain the highest time between the ranks and each other row is the time of a specific rank.

Naming of this file is temporary and will be changed.

The first time this main is called (i.e. if used with a script test/debug suite, only on the first call), also a .csv file with MPI rank allocations will be saved.

### `debug/` - Debugging `libswing`

The `debug/` directory builds an executable `bin/debug` that links with the `libswing.a` library and is used for debugging purposes. It provides a controlled environment to test and debug `libswing`'s functionality. This component is useful when trying to locate issues in the library's code.

**RIGHT NOW IT'S NOT USEFUL AND WILL BE COMPLETELY REWRITTEN.**

### `ompi_rules/` - Open MPI Rule File Generator

The `ompi_rules/` directory contains the source for a program that generates rule files for `Open MPI`. These rule files define the collective communication rules that Open MPI uses. The program compiles into the executable `bin/change_collective_rules`, which is used to select MPI Allreduce algorithm.

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

The project includes scripts to run the benchmark test suite and the debugging suite inside the `scripts/` directory.

The suggested workflow is to sbatch the `scripts/submit.sh` script when benchmarking (and when modified also for debugging) if on cluster, instead run the test\debug suite script on its own if running on local machine.

For now specific variables must be modified inside the run suites but it's planned to bring them on the submit script. Beware that everything for now it's extremely hand tailored on MY workflow and only now I'm polishing the program to let other people work with those scripts so a lot of modifications are still needed.


### `scripts/submit.sh` - Submit Tests via SLURM

For cluster-based environments that use SLURM for job scheduling, a template script `scripts/submit.sh` is provided. This script can be used to submit the test runs to the cluster via SLURM.

To use it, you will need to customize the script as per your SLURM job submission requirements. The script includes placeholders and options for specifying the number of processes, job time, and output location.

#### Example Usage for `submit.sh`

```bash
sbatch scripts/submit.sh
```
The script must be modified to select:
- `<p_name>`: Name of the partition to run tests on on the target cluster.
- `<qos_name>`: Name of the required quality of service. (OPTIONAL)
- `<n_nodes>`: The number of nodes to request for the SLURM job. Note that, for benchmarking reasons this must be the number of processes (i.e. one single process for node, irrespective of node cores).
- `<requested_time>`: The time requested for job allocation. Higher number of hours is suggested.
- `<account_name>`: Account name on the target cluster.
- `<LOCATION>`: Name of the machine, as defined in `scripts/environments/`.
- `<CUDA>`: If use CUDA-aware MPI. Beware that this option works only on Open MPI for now.
- `<OMPI_TEST>`: If use custom Open MPI library with swing implementations.

Also the standard output and error will be redirected into the results directory.

It will be modified to allow for algorithm selection directly in this stage. 

###### Warning
Beware that, especially with big allocations, those scripts can fail and waste compute hours if left uncheck. It's suggested, if possible, for big allocations, to check if the script starts working. If it does it's unlikely that compute time will be wasted. Currently working on the reliability of this part.

### `scripts/run_test_suite.sh` - Benchmarking Suite 

To run the benchmarking test suite without relying on the submit script, execute the following command:

```bash
scripts/run_test_suite.sh <num_processes>
```

- `<num_processes>`: The number of processes to use for the test run.

This script will set the necessary environmental variable based on the `location` variable.
To configure environments, add a new .sh script in `scripts/environments/`, export the necessary variables, and update the `location` variable to point to the new script.

Other two variables to set up in the script are `ompi_test` and `cuda`. The first one refers to the use of the modified Open MPI library with Swing Allreduce algorithms.

Important variables for the test selection are:
- `ALGO` selects the Allreduce algorithm. For now 0->default ompi selection, 1-7->specific ompi algorithm, 8-13->swing versions inside ompi, 14-16->swing over mpi.
- `ARR_SIZES` select the size of the test in number of elements in send buffers.
- `TYPES` select the datatype (or datatypes) to test.

At the end of the test, hostnames of nodes will be saved, but this functionality will be merged inside the `bin/test` executable to allow the mapping of MPI rank to actual node allocation (and not only guess it based on node number).

##### IMPORTANT
SKIP contains the algorithm to skip when `<num_processes> < <arr_size>` and must not be modified.

### `scripts/run_debug_suite.sh` - Debug Suite

Expect it to run like the test suite. Right now must be modified.

## Data Analysis and Visualization

Still in development. For now only two small scripts are added since I'm preparing everything to be used by more people.

Take a look, but it's still WIP.


## Building

To build the project, use the following steps:

1. **Build all components**:
   ```bash
   make
   ```

2. **Clean all builds**:
   ```bash
   make clean
   ```

The `Makefile` automatically handles the compilation and linking for each component, creating the necessary binaries and libraries in the `bin` and `lib` directories.

### Compiling Individual Parts of the Project
The project is built with a recursive makefile approach in which a top-level `Makefile` will run the `make` commands for each subfolder.

So, to build everything just run the `make` command in the main directory. You can also use the `make` command with the `clean` target to remove object files and binaries.

If you want to compile and build individual parts of the project you can either run the `make` command inside the desired subdirectory, or run `make -C <directory>`. This works also with the `clean` command.

## Results Management

In this repository, the results are stored in a directory that is automatically compressed into a `.tar` file to save storage space and keep the repository clean. The process works as follows:

1. **Results directory:** All raw results are saved in a dedicated directory, different for each system on which tests are run.
2. **Compression Script:** A script is included in the repository to compress the results directory into a `.tar` file. The script also add to `.gitignore` the uncompressed results subdirectory and remove possible duplicates from the `.gitignore` file.
3. **Pre-Commit Hook:** A pre-commit hook is set up to run the compression script automatically before each commit. This ensures that any new results are compressed and added to the repository in a consistent manner.

There is no need to run the script manually as everything is done automatically.

A `.csv` file will be added to contain results metadata.


## TODO
#### Makefile
- [ ] make the current static `libswing.a` a dynamic `libswing.so` to be added with `LD_PRELOAD`
- [ ] separate build process from testing\debugging
- [ ] add error handling when building\linking the library
#### Libswing modifications
- [ ] prepare for the possibility of implementing different collectives by refactoring code
- [ ] document functions and comment code
#### Test program
- [x] document functions and comment code
- [x] create a function to write rank and allocation inside test without relying on normal `srun` in test suite
- [ ] use enum when possible for clarity
- [ ] create a general interface to select specific testing for specific collectives without duplicating code by adding a sea of if-else statements (in particular modify the test loop to use a function pointer for each specific allreduce function and a switch for other collectives)
- [ ] separate and modularize ground truth check for different kinds of collectives
- [ ] standardize .csv format to what was decided with professor De Sensi
- [ ] create a function to add metadata in the .csv asked by the professor
#### OMPI rules
- [ ] change it so that it recognizes if the modified `Open MPI` is being run or not
- [ ] modify to let it work also with `MPICH` algorithm selection
#### Debug program
- [ ] WRITE EVERYTHING (it really sucks now)
#### Test/Debug suite Modifications
- [x] add error handling and more explicit messages about what is being done
- [x] separate results by system
- [x] comment the code
- [ ] build a better and clearer interface to select variables for testing
- [ ] separate common parts of `run_tests_scrips` and `run_debug_scrips` to separate subscripts to avoid code repetitions
- [ ] add an env var to select between `MPICH` and `Open MPI` binaries, independently of `ompi_test` (obviously `ompi_test` must be no when MPICH is selected)
#### Submit Script
- [x] automatically inject `$N_NODES` inside suite based on selected `-N` without further modifications
- [x] insert an `<output_directory>` for stderr and stdout of slurm
- [ ] bring variables to select inside test/debug suites to this layer in order to give a better interface. Modifications on those suite is needed when running test without the submit script
#### Results folder
- [x] add a script to compress the data
- [x] add pre-commit hook that triggers the compression of the new data, adds the uncompressed data to the gitignore
- [ ] add the script to build and update the .csv description
- [ ] add tests present on my systems
#### Plot python scripts
- [x] add the possibility of selecting specific tests
- [x] document and comment code
- [ ] avoid working only with summaries, or at least give the possibility of not doing it
- [ ] improve naming of output graphs so that they don't get overwritten by similar graphs
- [ ] refactor to separate graph drawing functions to separate files
