#include <stdio.h>

#include "test_tool.h"

int main(int argc, char *argv[]) {
  int rank, comm_sz, i;

  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  
  // Error checking for command-line arguments
  if (argc < 6) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <array_size> <iterations> <dtype> <rulepath> <dirpath>\n", argv[0]);
    }
    MPI_Abort(comm, 1);
  }

  char *endptr;
  size_t array_size = (size_t) strtoll(argv[1], &endptr, 10);
  if (*endptr != '\0' || array_size <= 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: Invalid array size. It must be a positive integer. Aborting...\n");
    }
    MPI_Abort(comm, 1);
  }

  int iter = (int) strtol(argv[2], &endptr, 10);
  if (*endptr != '\0' || iter <= 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: Invalid number of iterations. It must be a positive integer. Aborting...\n");
    }
    MPI_Abort(comm, 1);
  }

  const char* type_string = argv[3];
  MPI_Datatype dtype;
  size_t type_size;
  if (get_data_type(type_string, &dtype, &type_size) == -1){
    if (rank == 0){
      fprintf(stderr, "Error: Invalid datatype. Aborting...\n");
    }
    MPI_Abort(comm, 1);
  }
  
  // Allocate memory for the buffers
  char *sendbuf, *recvbuf, *recvbuf_gt;
  sendbuf = (char *)malloc(array_size * type_size);
  recvbuf = (char *)malloc(array_size * type_size);
  recvbuf_gt = (char *)malloc(array_size * type_size);
  if (sendbuf == NULL || recvbuf == NULL || recvbuf_gt == NULL) {
    if (rank == 0){
      fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    }
    MPI_Abort(comm, 1);
  } 
  
  if (rand_array_generator(sendbuf, type_string, array_size, rank) == -1){
    if (rank == 0){
      fprintf(stderr, "Error: sendbuf not generated correctly. Aborting...\n");
    }
    MPI_Abort(comm, 1);
  }

  // Each process will store its times for each iteration
  double start_time, end_time;
  double *times = (double *)malloc(iter * sizeof(double));

  // Perform ITER iterations of MPI_Allreduce and measure time
  for (i = 0; i < iter; i++) {
    MPI_Barrier(comm);
    start_time = MPI_Wtime();
    MPI_Allreduce(sendbuf, recvbuf, array_size, dtype, MPI_SUM, comm);
    end_time = MPI_Wtime();
    times[i] = end_time - start_time;
  }

  // Do a ground-truth check on the correctness of last iteration result
  MPI_Reduce(sendbuf, recvbuf_gt, array_size, dtype, MPI_SUM, 0, comm);
  MPI_Bcast(recvbuf_gt, array_size, dtype, 0, comm);
  if (are_equal(recvbuf, recvbuf_gt, array_size * type_size) == -1){
    if (rank == 0){
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
    }
    MPI_Abort(comm, 1);
  }

  // Gather all process times to rank 0
  double *all_times = NULL;
  if (rank == 0) {
    all_times = (double *)malloc(comm_sz * iter * sizeof(double));
  }
  MPI_Gather(times, iter, MPI_DOUBLE, all_times, iter, MPI_DOUBLE, 0, comm);

  // Find the highest execution time of each iteration
  double *highest = NULL;
  if (rank == 0) {
    highest = (double *)malloc(iter * sizeof(double));
  }
  MPI_Reduce(times, highest, iter, MPI_DOUBLE, MPI_MAX, 0, comm);

  char filename[256];
  const char *rulepath = argv[4];
  if (create_filename(filename, sizeof(filename), comm_sz, array_size, type_string, rulepath) == -1) {
    fprintf(stderr, "Error: Failed to create the filename.\n");
    MPI_Abort(comm, 1);
  }

  const char *dirpath = argv[5];
  char fullpath[MAX_PATH_LENGTH];
  if (concatenate_path(dirpath, filename, fullpath) == -1) {
    fprintf(stderr, "Error: Failed to create fullpath.\n");
    MPI_Abort(comm, 1);
  }

  FILE *output_file = NULL;
  if (rank == 0) {
    output_file = fopen(fullpath, "w");
    if (output_file == NULL) {
      fprintf(stderr, "Error: Opening file %s for writing\n", fullpath);
      MPI_Abort(comm, 1);
    }

    fprintf(output_file, "# highest, rank1, rank2, ..., rankn (time is in ns (i.e. 10^-9 s))\n");
    for (i = 0; i < iter; i++) {
      fprintf(output_file, "%d", (int) (highest[i] * 1000000000)); // Write iteration number and the highest number
      for (int j = 0; j < comm_sz; j++) {
        fprintf(output_file, ", %d", (int) (all_times[j * iter + i] * 1000000000)); // Write the time for each rank
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
    free(highest);
  }
  
  MPI_Finalize();
  return 0;
}

