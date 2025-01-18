#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_tool.h"
#include "libswing.h"

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  // Declare beforehand the buffers to allow for correct `goto` error handling
  char *sendbuf = NULL, *recvbuf = NULL, *recvbuf_gt = NULL;
  double *times = NULL, *all_times = NULL, *highest = NULL;
  double start_time, end_time;
  
  size_t array_size;
  int iter, alg_number;
  const char* type_string, *outputdir;
  // Get command line arguments
  if (get_command_line_arguments(argc, argv, &array_size, &iter, &type_string,
                                 &alg_number, &outputdir) == -1){
    goto cleanup;
  }
  
  // Get size and MPI_Datatype from input `type_string`
  MPI_Datatype dtype;
  size_t type_size;
  if (get_data_type(type_string, &dtype, &type_size) == -1){
    goto cleanup;
  }

  sendbuf = (char *)malloc(array_size * type_size);
  recvbuf = (char *)malloc(array_size * type_size);
  recvbuf_gt = (char *)malloc(array_size * type_size);
  times = (double *)malloc(iter * sizeof(double));
  
  // Allocate memory for all ranks
  if (sendbuf == NULL || recvbuf == NULL || recvbuf_gt == NULL || times == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    goto cleanup;
  }

  // Allocate memory for rank0-specific buffers
  if (rank == 0) {
    all_times = (double *)malloc(comm_sz * iter * sizeof(double));
    highest = (double *)malloc(iter * sizeof(double));
    if (all_times == NULL || highest == NULL){
      fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
      goto cleanup;
    }
  }
  
  // randomly generate the sendbuf
  if (rand_array_generator(sendbuf, type_string, array_size, rank) == -1){
    goto cleanup;
  }

  // Perform ITER iterations of MPI_Allreduce and measure time
  // TODO: use a switch, a function pointer or whatever else, but this is temporary
  for (int i = 0; i < iter; i++) {
    MPI_Barrier(comm);
    start_time = MPI_Wtime();
    if (alg_number == 14){
      allreduce_recursivedoubling(sendbuf, recvbuf, array_size, dtype, MPI_SUM, comm);
    }
    else if (alg_number == 15){
      allreduce_swing_lat(sendbuf, recvbuf, array_size, dtype, MPI_SUM, comm);
    }
    else if (alg_number == 16){
      allreduce_swing_bdw_static(sendbuf, recvbuf, array_size, dtype, MPI_SUM, comm);
    }
    else {
      MPI_Allreduce(sendbuf, recvbuf, array_size, dtype, MPI_SUM, comm);
    }
    end_time = MPI_Wtime();
    times[i] = end_time - start_time;
  }

  // Do a ground-truth check on the correctness of last iteration result
  MPI_Reduce(sendbuf, recvbuf_gt, array_size, dtype, MPI_SUM, 0, comm);
  MPI_Bcast(recvbuf_gt, array_size, dtype, 0, comm);
  if(dtype != MPI_DOUBLE && dtype != MPI_FLOAT){
    if (memcmp(recvbuf, recvbuf_gt, array_size * type_size) != 0){
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      goto cleanup;
    }
  } else{
    // On floating point arithmetic ground-truth check also consider rounding errors
    if (are_equal_eps(recvbuf_gt, recvbuf, array_size, type_string, comm_sz) == -1){
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      goto cleanup;
    }
  }

  // Gather all process times to rank 0 and find the highest execution time of each iteration
  MPI_Gather(times, iter, MPI_DOUBLE, all_times, iter, MPI_DOUBLE, 0, comm);
  MPI_Reduce(times, highest, iter, MPI_DOUBLE, MPI_MAX, 0, comm);
  
  // Save results to a .csv file inside `/data/` subdirectory. Bash script `run_test_suite.sh`
  // is responsible to create the `/data/` subdir.
  if (rank == 0){
    char data_filename[128], data_fullpath[TEST_MAX_PATH_LENGTH];
    snprintf(data_filename, sizeof(data_filename), "data/%d_%ld_%s_%d.csv",
             comm_sz, array_size, type_string, alg_number);
    if (concatenate_path(outputdir, data_filename, data_fullpath) == -1) {
      fprintf(stderr, "Error: Failed to create fullpath.\n");
      goto cleanup;
    }
    if (write_output_to_file(data_fullpath, highest, all_times, iter, comm_sz) == -1){
      goto cleanup;
    }
  }

  // Save to file allocations (it uses MPI parallel I/O operations)
  char alloc_filename[128] = "alloc.csv";
  char alloc_fullpath[TEST_MAX_PATH_LENGTH];
  if (concatenate_path(outputdir, alloc_filename, alloc_fullpath) == -1){
    fprintf(stderr, "Error: Failed to create alloc_fullpath.\n");
    goto cleanup;
  }

  // Write current allocations if and only if the file `alloc_fullpath`
  // does not exists (i.e. only the first time for each srun this function
  // is encountered the allocations will be registerd)
  int should_write_alloc;
  if (rank == 0){
    should_write_alloc = file_not_exists(alloc_fullpath);
  }
  MPI_Bcast(&should_write_alloc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (should_write_alloc == 1 && write_allocations_to_file(alloc_fullpath) == -1){
    goto cleanup;
  }

  // Clean up
  free(sendbuf);
  free(recvbuf);
  free(recvbuf_gt);
  free(times);
  if (rank == 0){
    free(all_times);
    free(highest);
  }
  
  MPI_Finalize();

  return EXIT_SUCCESS;

cleanup:
  if (NULL != sendbuf)    free(sendbuf);
  if (NULL != recvbuf)    free(recvbuf);
  if (NULL != recvbuf_gt) free(recvbuf_gt);
  if (NULL != times)      free(times);
  
  if (rank == 0) {
    if (NULL != all_times)  free(all_times);
    if (NULL != highest)    free(highest);
  }

  MPI_Finalize();

  return EXIT_FAILURE;
}

