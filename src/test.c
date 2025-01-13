#include "test_tool.h"
#include "libswing.h"

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  
  
  // Allocate memory for the buffers
  char *sendbuf = NULL, *recvbuf = NULL, *recvbuf_gt = NULL;
  double *times = NULL, *all_times = NULL, *highest = NULL;
  double start_time, end_time;
  
  size_t array_size;
  int iter, alg_number;
  const char* type_string, *dirpath;
  if (get_command_line_arguments(argc, argv, &array_size, &iter, &type_string, &alg_number, &dirpath) == -1){
    goto cleanup;
  }

  MPI_Datatype dtype;
  size_t type_size;
  if (get_data_type(type_string, &dtype, &type_size) == -1){
    goto cleanup;
  }

  sendbuf = (char *)malloc(array_size * type_size);
  recvbuf = (char *)malloc(array_size * type_size);
  recvbuf_gt = (char *)malloc(array_size * type_size);
  times = (double *)malloc(iter * sizeof(double));
  
  if (sendbuf == NULL || recvbuf == NULL || recvbuf_gt == NULL || times == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    goto cleanup;
  }

  // Allocate memory for root-specific buffers
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
  for (int i = 0; i < iter; i++) {
    if (alg_number == 14){
      MPI_Barrier(comm);
      start_time = MPI_Wtime();
      allreduce_recursivedoubling(sendbuf, recvbuf, array_size, dtype, MPI_SUM, comm);
      end_time = MPI_Wtime();
      times[i] = end_time - start_time;
    }
    else if (alg_number == 15){
      MPI_Barrier(comm);
      start_time = MPI_Wtime();
      allreduce_swing_lat(sendbuf, recvbuf, array_size, dtype, MPI_SUM, comm);
      end_time = MPI_Wtime();
      times[i] = end_time - start_time;
    }
    else {
      MPI_Barrier(comm);
      start_time = MPI_Wtime();
      MPI_Allreduce(sendbuf, recvbuf, array_size, dtype, MPI_SUM, comm);
      end_time = MPI_Wtime();
      times[i] = end_time - start_time;
    }
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
  
  // Save everything to file
  if (rank == 0){
    char filename[256], fullpath[MAX_PATH_LENGTH];
    snprintf(filename, sizeof(filename), "%d_%ld_%s_%d.csv", comm_sz, array_size, type_string, alg_number);
    if (concatenate_path(dirpath, filename, fullpath) == -1) {
      fprintf(stderr, "Error: Failed to create fullpath.\n");
      goto cleanup;
    }
    if (write_output_to_file(fullpath, highest, all_times, iter, comm_sz) == -1){
      goto cleanup;
    }
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

