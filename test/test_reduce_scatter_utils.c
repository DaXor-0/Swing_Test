#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "libswing.h"
#include "test_utils.h"

int reduce_scatter_allocator(void **sbuf, void **rbuf, void **rbuf_gt, size_t count,
                             size_t type_size, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  
  // sbuf must contain only the data specific to the current rank,
  // while rbuf (and rbuf_gt) must contain the data from all ranks.
  *sbuf = (char *)malloc(count * type_size);
  *rbuf = (char *)malloc((count / (size_t) comm_sz) * type_size);
  *rbuf_gt = (char *)malloc((count / (size_t) comm_sz) * type_size);
  if (*sbuf == NULL || *rbuf == NULL || *rbuf_gt == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    return -1;
  }
  return 0; // Success
}

int reduce_scatter_gt_check(REDUCE_SCATTER_ARGS, void *rbuf_gt) {
  // Compute the ground-truth result using PMPI_Reduce_scatter.
  PMPI_Reduce_scatter(sbuf, rbuf_gt, rcounts, dtype, op, comm);

  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  
  int type_size;
  MPI_Type_size(dtype, &type_size);

  if (dtype != MPI_DOUBLE && dtype != MPI_FLOAT) {
    // For non-floating-point types, use memcmp for exact comparison.
    if (memcmp(rbuf, rbuf_gt, rcounts[rank] * type_size) != 0) {
      #ifdef DEBUG
      debug_print_buffers(rbuf, rbuf_gt, rcounts[rank], dtype, comm);
      #endif
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  } else {
    // For floating-point types, use an epsilon-based comparison to account for rounding errors.
    if (are_equal_eps(rbuf_gt, rbuf, rcounts[rank], dtype, comm_sz) == -1) {
      #ifdef DEBUG
      debug_print_buffers(rbuf, rbuf_gt, rcounts[rank], dtype, comm);
      #endif
      fprintf(stderr, "Error: results are not valid. Aborting...\n");
      return -1;
    }
  }

  return 0; // Success.
}

/**
 * @brief Selects the appropriate reduce scatter algorithm based on an algorithm number.
 * 
 * This function returns a pointer to the appropriate reduce scatter function based on the
 * provided algorithm enum. If the algorithm number does not match any custom implementation,
 * it returns a pointer to the MPI_Reduce_scatter function by default.
 * 
 * @param algorithm The algorithm enum specifying which allreduce function to use.
 * 
 * @return Pointer to the selected allreduce function.
 *
 * WARNING: This function does not check if the algorithm number is valid. It is the caller's
 * responsibility
 */
static inline reduce_scatter_func_ptr get_reduce_scatter_function(reduce_scatter_algo_t algorithm) {
  switch (algorithm) {
    case REDUCE_SCATTER_RECURSIVE_HALVING_OVER:
      return reduce_scatter_recursivehalving;
    case REDUCE_SCATTER_RING_OVER:
      return reduce_scatter_ring;
    case REDUCE_SCATTER_BUTTERFLY_OVER:
      return reduce_scatter_butterfly;
    default:
      return MPI_Reduce_scatter;
  }
}


void reduce_scatter_test_loop(REDUCE_SCATTER_ARGS, int iter, double *times, reduce_scatter_algo_t algorithm){
  reduce_scatter_func_ptr reduce_scatter_func = get_reduce_scatter_function(algorithm);

  double start_time, end_time;

  for (int i = 0; i < iter; i++) {
    MPI_Barrier(comm);
    start_time = MPI_Wtime();
    reduce_scatter_func(sbuf, rbuf, rcounts, dtype, op, comm);
    end_time = MPI_Wtime();
    times[i] = end_time - start_time;
  }
}
