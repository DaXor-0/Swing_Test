#include "libswing.h"
#include <math.h>
#include <mpi.h>

#define SWING_MAX_STEPS 20 ///< Maximum number of steps in the swing algorithm. Supports up to 2^20 nodes.

static int rhos[SWING_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};

/**
 * @brief Computes the destination rank for a given process in a swing algorithm step.
 *
 * This function calculates the rank to which a process will communicate based on the swing algorithm,
 * ensuring the result is within the valid range of ranks.
 *
 * @param rank The rank of the current process.
 * @param step The current step in the swing algorithm.
 * @param comm_sz The total number of processes in the communicator.
 * @return The destination rank after applying the swing algorithm, a value in [0, comm_sz - 1].
 */
static inline int pi(int rank, int step, int comm_sz) {
  int dest;

  if ((rank & 1) == 0) dest = (rank + rhos[step]) % comm_sz;  // Even rank
  else dest = (rank - rhos[step]) % comm_sz;                  // Odd rank

  if (dest < 0) dest += comm_sz;                              // Adjust for negative ranks

  return dest;
}

/**
 * @brief Copies data from an input buffer to an output buffer.
 *
 * This function validates the input parameters and performs a memory copy of `count` elements
 * from the input buffer to the output buffer, using the size of the specified MPI datatype.
 *
 * @param input_buffer Pointer to the source buffer.
 * @param output_buffer Pointer to the destination buffer.
 * @param count Number of elements to copy.
 * @param datatype The MPI datatype of each element.
 * @return 0 on success, or -1 if the input parameters are invalid.
 */
static inline int copy_buffer(const void *input_buffer, void *output_buffer, size_t count, const MPI_Datatype datatype) {
  if (input_buffer == NULL || output_buffer == NULL || count <= 0) {
    return -1; ///< Invalid input parameters
  }

  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);                // Get the size of the MPI datatype

  size_t total_size = count * (size_t)datatype_size;

  memcpy(output_buffer, input_buffer, total_size);        // Perform the memory copy

  return 0;
}

/**
 * @brief Computes the memory span for `count` repetitions of the given MPI datatype.
 *
 * This function calculates the total memory required for `count` repetitions of an MPI datatype,
 * including the gap at the beginning (true lower bound) and excluding padding at the end.
 *
 * @param datatype The MPI datatype.
 * @param count Number of repetitions of the datatype.
 * @param gap Pointer to store the gap (true lower bound) at the beginning.
 * @return The total memory span required for `count` repetitions of the datatype.
 */
static inline ptrdiff_t datatype_span(MPI_Datatype datatype, size_t count, ptrdiff_t *gap) {
  if (count == 0) {
    *gap = 0;
    return 0;                                                  // No memory span required for zero repetitions
  }

  MPI_Aint lb, extent;
  MPI_Aint true_lb, true_extent;

  MPI_Type_get_extent(datatype, &lb, &extent);                // Get the extent of the datatype
  MPI_Type_get_true_extent(datatype, &true_lb, &true_extent); // Get the true extent of the datatype

  *gap = true_lb;                                             // Store the true lower bound

  return true_extent + extent * (count - 1);                  // Calculate the total memory span
}

/**
 * @brief Finds the largest power of 2 less than or equal to the given size.
 *
 * This function uses bitwise operations to compute the largest power of 2 that is less than
 * or equal to the input size, in O(log(size)) time.
 *
 * @param size The input value.
 * @return The largest power of 2 less than or equal to the input size, or -1 if the input size is non-positive.
 */
static inline int nearest_po2(int size) {
  if (size <= 0) return -1;

  size |= size >> 1;
  size |= size >> 2;
  size |= size >> 4;
  size |= size >> 8;
  size |= size >> 16;

  return size - (size >> 1);
}

static inline int mylog2(int x) {
    return sizeof(int)*8 - 1 - __builtin_clz(x);
}

// Function to perform the addition operation
static inline void reduction(int64_t *buf1, int64_t *buf2, size_t count) {
  for (size_t i = 0; i < count; ++i) {
      buf2[i] += buf1[i];
  }
}

int allreduce_swing_lat(const void *sbuf, void *rbuf, size_t count, MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) {
  int rank, size;
  int ret, line; // for error handling
  char *tmpsend, *tmprecv, *inplacebuf_free = NULL;
  ptrdiff_t span, gap = 0;
  

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Special case for size == 1
  if (1 == size) {
    if (MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }
  
  // Allocate and initialize temporary send buffer
  span = datatype_span(dtype, count, &gap);
  inplacebuf_free = (char*) malloc(span + gap);
  char *inplacebuf = inplacebuf_free + gap;

  // Copy content from sbuffer to inplacebuf
  if (MPI_IN_PLACE == sbuf) {
      ret = copy_buffer((char*)rbuf, inplacebuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  } else {
      ret = copy_buffer((char*)sbuf, inplacebuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  }

  tmpsend = inplacebuf;
  tmprecv = (char*) rbuf;
  
  int extra_ranks, new_rank = rank, loop_flag = 0; // needed to handle not power of 2 cases

  //Determine nearest power of two less than or equal to size
  int adjsize = nearest_po2(size);
  if (-1 == adjsize) {
    return MPI_ERR_ARG;
  }
  int steps = mylog2(adjsize);

  //Number of nodes that exceed max(2^n)< size
  extra_ranks = size - adjsize;
  int is_power_of_two = size >> 1 == adjsize;


  // First part of computation to get a 2^n number of nodes.
  // What happens is that first #extra_rank even nodes sends their
  // data to the successive node and do not partecipate in the general
  // collective call operation.
  // All the nodes that do not stop their computation will receive an alias
  // called new_node, used to calculate their correct destination wrt this
  // new "cut" topology.
  if (rank <  (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret = MPI_Send(tmpsend, count, dtype, (rank + 1), 0, comm);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      loop_flag = 1;
    } else {
      ret = MPI_Recv(tmprecv, count, dtype, (rank - 1), 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
      new_rank = rank >> 1;
    }
  } else new_rank = rank - extra_ranks;
  
  
  // Actual allreduce computation for general cases
  int s, vdest, dest;
  for (s = 0; s < steps; s++){
    if (loop_flag) break;
    vdest = pi(new_rank, s, adjsize);

    dest = is_power_of_two ?
              vdest :
              (vdest < extra_ranks) ?
              (vdest << 1) + 1 : vdest + extra_ranks;

    ret = MPI_Sendrecv(tmpsend, count, dtype, dest, 0,
                       tmprecv, count, dtype, dest, 0,
                       comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    
    MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
  }
  
  // Final results is sent to nodes that are not included in general computation
  // (general computation loop requires 2^n nodes).
  if (rank < (2 * extra_ranks)){
    if (!loop_flag){
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    } else {
      ret = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      tmpsend = (char*)rbuf;
    }
  }

  if (tmpsend != rbuf) {
    ret = copy_buffer(tmpsend, (char*) rbuf, count, dtype);
    if (ret < 0) { line = __LINE__; goto error_hndl; }
  }

  free(inplacebuf_free);
  return MPI_SUCCESS;

  error_hndl:
    fprintf(stderr, "%s:%4d\tRank %d Error occurred %d\n", __FILE__, line, rank, ret);
    (void)line;  // silence compiler warning
    if (NULL != inplacebuf_free) free(inplacebuf_free);
    return ret;
}


int allreduce_recursivedoubling(const void *sbuf, void *rbuf, size_t count,
                                MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
{
  int ret, line, rank, size, adjsize, remote, distance;
  int newrank, newremote, extra_ranks;
  char *tmpsend = NULL, *tmprecv = NULL, *inplacebuf_free = NULL, *inplacebuf;
  ptrdiff_t span, gap = 0;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /* Special case for size == 1 */
  if (1 == size) {
    if (MPI_IN_PLACE != sbuf) {
      ret = copy_buffer((char *) sbuf, (char *) rbuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }

  /* Allocate and initialize temporary send buffer */
  span = datatype_span(dtype, count, &gap);

  inplacebuf_free = (char*) malloc(span);
  if (NULL == inplacebuf_free) { ret = -1; line = __LINE__; goto error_hndl; }
  inplacebuf = inplacebuf_free - gap;

  if (MPI_IN_PLACE == sbuf) {
      ret = copy_buffer((char*)rbuf, inplacebuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  } else {
      ret = copy_buffer((char*)sbuf, inplacebuf, count, dtype);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  }

  tmpsend = (char*) inplacebuf;
  tmprecv = (char*) rbuf;

  /* Determine nearest power of two less than or equal to size */
  adjsize = nearest_po2(size);

  /* Handle non-power-of-two case:
     - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
     sets new rank to -1.
     - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
     apply appropriate operation, and set new rank to rank/2
     - Everyone else sets rank to rank - extra_ranks
  */
  extra_ranks = size - adjsize;
  if (rank <  (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret = MPI_Send(tmpsend, count, dtype, (rank + 1), 0, comm);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      newrank = -1;
    } else {
      ret = MPI_Recv(tmprecv, count, dtype, (rank - 1), 0, comm,
                  MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      /* tmpsend = tmprecv (op) tmpsend */
      // reduction((int64_t *) tmprecv, (int64_t *) tmpsend, count);
      MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
      newrank = rank >> 1;
    }
  } else {
    newrank = rank - extra_ranks;
  }

  /* Communication/Computation loop
     - Exchange message with remote node.
     - Perform appropriate operation taking in account order of operations:
     result = value (op) result
  */
  for (distance = 0x1; distance < adjsize; distance <<=1) {
    if (newrank < 0) break;
    /* Determine remote node */
    newremote = newrank ^ distance;
    remote = (newremote < extra_ranks) ? (newremote * 2 + 1) : (newremote + extra_ranks);
    
    /* Exchange the data */
    ret = MPI_Sendrecv(tmpsend, count, dtype, remote, 0,
                       tmprecv, count, dtype, remote, 0,
                       comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    // reduction((int64_t *) tmprecv, (int64_t *) tmpsend, count);
    MPI_Reduce_local((char *) tmprecv, (char *) tmpsend, count, dtype, op);
  }

  /* Handle non-power-of-two case:
     - Odd ranks less than 2 * extra_ranks send result from tmpsend to
     (rank - 1)
     - Even ranks less than 2 * extra_ranks receive result from (rank + 1)
  */
  if (rank < (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret = MPI_Recv(rbuf, count, dtype, (rank + 1), 0, comm, MPI_STATUS_IGNORE);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      tmpsend = (char*)rbuf;
    } else {
      ret = MPI_Send(tmpsend, count, dtype, (rank - 1), 0, comm);
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    }
  }

  /* Ensure that the final result is in rbuf */
  if (tmpsend != rbuf) {
    ret = copy_buffer(tmpsend, (char*)rbuf, count, dtype);
    if (ret < 0) { line = __LINE__; goto error_hndl; }
  }

  if (NULL != inplacebuf_free) free(inplacebuf_free);
  return MPI_SUCCESS;

  error_hndl:
    fprintf(stderr, "%s:%4d\tRank %d Error occurred %d\n", __FILE__, line, rank, ret);
    (void)line;  // silence compiler warning
    if (NULL != inplacebuf_free) free(inplacebuf_free);
    return ret;
}

