# Project Overview

This project consists of several components, including a static library (`libswing`), a benchmark for the library (`test`), a debug version of the library (`debug`), and a set of rules for Open MPI (`ompi_rules`). Below is a description of each component and how to build and run them.

## Components

### `libswing` - Static Library

The `libswing` directory contains the source code for the `libswing` static library. This library is compiled into a static archive (`libswing.a`) and is used by the other components in the project. It provides essential utilities and functions that are benchmarked and debugged in the subsequent components.

### `test` - Benchmark for `libswing`

The `test` directory contains a set of benchmark tests that are used to measure the performance of the `libswing` library. It compiles an executable that runs various tests and benchmarks using the functionality provided by `libswing`.

### `debug` - Debugging `libswing`

The `debug` directory builds an executable that links with the `libswing` library and is used for debugging purposes. It provides a controlled environment to test and debug `libswing`'s functionality. This component is useful when trying to locate issues in the library's code.

### `ompi_rules` - Open MPI Rule File Generator

The `ompi_rules` directory contains the source for a program that generates rule files for Open MPI. These rule files define the collective communication rules that Open MPI uses. The program compiles into an executable called `change_collective_rules`, which can be used to create and modify these rules for different MPI configurations.

## Running Tests

The project includes a script to run the test suite (`run_test_suite.sh`). The script runs the `test` executable with different settings, depending on the environment (local machine, cluster, or others). Before running the script, you need to specify how many processes to run and where to save the results.

### Running the Test Suite

To run the test suite, execute the following command:

```bash
./run_test_suite.sh <num_processes> <output_directory>
```

- `<num_processes>`: The number of processes to use for the test run.
- `<output_directory>`: The directory where the test results will be saved.

This script will automatically select the correct configuration based on the environment (cluster, local, or others).

### Submit Tests via SLURM

For cluster-based environments that use SLURM for job scheduling, a template script `submit.sh` is provided. This script can be used to submit the test runs to the cluster via SLURM.

To use it, you will need to customize the script as per your SLURM job submission requirements. The script includes placeholders and options for specifying the number of processes, job time, and output location.

### Example Usage for `submit.sh`

```bash
sbatch submit.sh <num_processes> <output_directory>
```

- `<num_processes>`: The number of processes to request for the SLURM job.
- `<output_directory>`: The directory where the results of the job will be stored.

## Building the Project

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

## Compiling Individual Parts of the Project

You can compile and build individual parts of the project separately using the `make` command in their respective directories. Below are the steps to compile each component:

### 1. Compile the `libswing` Library

To compile the `libswing` static library, navigate to the `libswing` directory and run:

```bash
make -C libswing
```

This will compile the library and create the static archive (`libswing.a`) in the `lib` directory.

### 2. Compile the `debug` Executable

To compile the `debug` executable, which links with the `libswing` library for debugging purposes, run:

```bash
make -C debug
```

This will create the `debug` executable in the `bin` directory, which you can use for debugging.

### 3. Compile the `test` Executable

To compile the `test` executable (the benchmark for `libswing`), navigate to the `test` directory and run:

```bash
make -C test
```

This will compile the `test` executable and place it in the `bin` directory.

### 4. Compile the `ompi_rules` Executable

To compile the executable for generating Open MPI rule files, run:

```bash
make -C ompi_rules
```

This will generate the `change_collective_rules` executable in the `bin` directory.

---

You can also use the `make` command with the `clean` target to remove object files and binaries after compiling individual components:

```bash
make -C <directory> clean
```

This will clean the build artifacts in the specified directory (e.g., `libswing`, `test`, `debug`, or `ompi_rules`).

### Directory Structure

```
.
├── bin                   # Binaries (executables)
├── debug                 # Debug version of libswing
│   └── debug.c           # Debug source code
├── include               # Header files
│   └── libswing.h        # Main library header
├── lib                   # Static library for libswing
│   └── libswing.a        # Compiled static library
├── libswing              # Source code for the static library
│   ├── libswing.c        # Main library code
│   └── libswing_bitmaps.c # Bitmaps handling for the library
├── makefile              # Top-level Makefile
├── obj                   # Object files
└── test                  # Benchmark test source code
│   ├── test.c            # Main test code
│   └── test_tool.c       # Additional test tools
├── ompi_rules            # Open MPI rule file generator
│   └── change_collective_rules.c # Rule file generator source code
└── run_test_suite.sh     # Script to run the test suite
└── submit.sh             # SLURM job submission script
```

<!-- ## Dependencies -->
<!---->
<!-- - `mpicc` (MPI C Compiler) -->
<!-- - `open MPI` -->
<!-- - `gcc` (GNU Compiler Collection) -->
<!---->
<!-- Ensure that your environment is set up with the necessary compilers and MPI libraries to successfully compile and run the components. -->
<!---->
<!-- ## License -->
<!---->
<!-- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->
