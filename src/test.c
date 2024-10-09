#include <stdio.h>
#include <mpi.h>

#include "test_tool.h"

#define ITER 50

int main(int argc, char *argv[]) {
  int rank, comm_sz, i;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  
  // Error checking for command-line arguments
  if (argc < 2) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <array_size>\n", argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  char *endptr;
  int errno = 0;  // To distinguish success/failure after the call
  size_t array_size = (size_t) strtoll(argv[1], &endptr, 10);
  // Error checking for strtoll
  if (errno != 0 || *endptr != '\0' || array_size <= 0) {
    if (rank == 0) {
      fprintf(stderr, "Invalid array size. It must be a positive integer.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Allocate memory for the buffers
  int *sendbuf, *recvbuf, *recvbuf_gt;
  sendbuf = (int *)malloc(array_size * sizeof(int));
  recvbuf = (int *)malloc(array_size * sizeof(int));
  recvbuf_gt = (int *)malloc(array_size * sizeof(int));
  if (sendbuf == NULL || recvbuf == NULL || recvbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  } 
  
  rand_array_generator(sendbuf, array_size, rank);

  // Each process will store its times for each iteration
  double start_time, end_time;
  double *times = (double *)malloc(ITER * sizeof(double));

  // Perform ITER iterations of MPI_Allreduce and measure time
  for (i = 0; i < ITER; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    MPI_Allreduce(sendbuf, recvbuf, array_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    times[i] = end_time - start_time;
  }

  // Do a ground-truth check on the correctness of last iteration result
  MPI_Reduce(sendbuf, recvbuf_gt, array_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(recvbuf_gt, array_size, MPI_INT, 0, MPI_COMM_WORLD);
  int is_valid = are_equal(recvbuf, recvbuf_gt, array_size * sizeof(recvbuf[0]));
  int global_is_valid;
  MPI_Reduce(&is_valid, &global_is_valid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  // Gather all process times to rank 0
  double *all_times = NULL;
  if (rank == 0) {
    all_times = (double *)malloc(comm_sz * ITER * sizeof(double));
  }
  MPI_Gather(times, ITER, MPI_DOUBLE, all_times, ITER, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  
  char filename[256];
  if (create_filename(filename, sizeof(filename), comm_sz, array_size) == -1) {
      fprintf(stderr, "Error: Failed to create the filename.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

  FILE *output_file = NULL;
  if (rank == 0) {
    output_file = fopen(filename, "w");
    if (output_file == NULL) {
      fprintf(stderr, "Error opening file %s for writing\n", filename);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fprintf(output_file, "IS VALID : %d\niter, rank1, rank2, ..., rankn\n", global_is_valid);
    for (i = 0; i < ITER; i++) {
      fprintf(output_file, "%d", i + 1); // Write iteration number
      for (int j = 0; j < comm_sz; j++) {
        fprintf(output_file, " %.6f", all_times[j * ITER + i]); // Write the time for each rank
      }
      fprintf(output_file, "\n");
    }
    fclose(output_file);
  }

  // Clean up
  free(sendbuf);
  free(recvbuf);
  free(recvbuf_gt);
  free(times);
  if (rank == 0){
    free(all_times);
  }
  
  MPI_Finalize();
  return 0;
}
