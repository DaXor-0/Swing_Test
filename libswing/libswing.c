#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"
#include "libswing_bitmaps.h"

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
  
  // Determine nearest power of two less than or equal to size
  // and return an error if size is 0
  int steps = hibit(size, (int)(sizeof(size) * CHAR_BIT) - 1);
  if (steps == -1) {
      return MPI_ERR_ARG;
  }
  int adjsize = 1 << steps;  // Largest power of two <= size

  // Number of nodes that exceed the largest power of two less than or equal to size
  int extra_ranks = size - adjsize;
  int is_power_of_two = (size & (size - 1)) == 0;


  // First part of computation to get a 2^n number of nodes.
  // What happens is that first #extra_rank even nodes sends their
  // data to the successive node and do not partecipate in the general
  // collective call operation.
  // All the nodes that do not stop their computation will receive an alias
  // called new_node, used to calculate their correct destination wrt this
  // new "cut" topology.
  int new_rank = rank, loop_flag = 0;
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
  adjsize = next_poweroftwo(size);

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



int allreduce_swing_bdw_static(const void *send_buf, void *recv_buf, size_t count,
                               MPI_Datatype dtype, MPI_Op op, MPI_Comm comm){
  int size, rank; 
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to size,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if size != adj_size
  //Determine nearest power of two less than or equal to size
  int steps = hibit(size, (int) (sizeof(size) * CHAR_BIT) - 1);
  if (-1 == steps){
    return MPI_ERR_ARG;
  }
  int adjsize = 1 << steps;
  
  //WARNING: Assuming size is a pow of 2
  int vrank, vdest;
  vrank = rank;
  
  ptrdiff_t lb, extent, gap = 0;
  MPI_Type_get_extent(dtype, &lb, &extent);
  
  int split_rank;
  size_t small_blocks, big_blocks;
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, adjsize, split_rank,
                               big_blocks, small_blocks);
  
  // Find the biggest power-of-two smaller than count to allocate
  // as few memory as necessary for buffers
  int n_pow = hibit((int) count, (int) (sizeof(count) * CHAR_BIT) -1); 
  size_t buf_count = 1 << n_pow;
  ptrdiff_t buf_size = datatype_span(dtype, buf_count, &gap);

  // Temporary target buffer for send operations and source buffer
  // for reduce and overwrite operations
  char *tmp_send = NULL, *tmp_recv = NULL;
  char *tmp_buf_raw = NULL, *tmp_buf;
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;

  // Copy into receive_buffer content of send_buffer to not produce
  // side effects on send_buffer
  if (send_buf != MPI_IN_PLACE) {
    copy_buffer((char *)recv_buf, (char *)send_buf, count, dtype);
  }
  
  const int *s_bitmap = NULL, *r_bitmap = NULL;
  if(get_static_bitmap(&s_bitmap, &r_bitmap, steps, size, rank) == -1){
    free(tmp_buf);
    return MPI_ERR_UNKNOWN;
  }
  
  int step, w_size = adjsize;
  size_t s_count, r_count;
  ptrdiff_t s_offset, r_offset;
  // Reduce-Scatter phase
  for (step = 0; step < steps; step++) {
    w_size >>= 1;
    vdest = pi(vrank, step, adjsize);

    s_count = (s_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (s_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - s_bitmap[step]);
    s_offset = (s_bitmap[step] <= split_rank) ?
                (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(s_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;

    r_count = (r_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (r_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - r_bitmap[step]);
    r_offset = (r_bitmap[step] <= split_rank) ?
                (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(r_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;
    
    tmp_send = (char *)recv_buf + s_offset;
    MPI_Sendrecv(tmp_send, s_count, dtype, vdest, 0,
                 tmp_buf, r_count, dtype, vdest, 0,
                 comm, MPI_STATUS_IGNORE);
    
    tmp_recv = (char *) recv_buf + r_offset;
    MPI_Reduce_local(tmp_buf, tmp_recv, r_count, dtype, op);
  }
  
  // Allgather phase
  for(step = steps - 1; step >= 0; step--) {
    vdest = pi(vrank, step, adjsize);
    
    s_count = (s_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (s_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - s_bitmap[step]);
    s_offset = (s_bitmap[step] <= split_rank) ?
                (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(s_bitmap[step] * (int) small_blocks + split_rank) * (ptrdiff_t) extent;

    r_count = (r_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * big_blocks :
                  (r_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_blocks :
                    (size_t)w_size * small_blocks + (size_t)(split_rank - r_bitmap[step]);
    r_offset = (r_bitmap[step] <= split_rank) ?
                (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(big_blocks * extent) :
                (ptrdiff_t)(r_bitmap[step] * (int)small_blocks + split_rank) * (ptrdiff_t) extent;
    
    tmp_send = (char *)recv_buf + s_offset;
    tmp_recv = (char *)recv_buf + r_offset;

    MPI_Sendrecv(tmp_recv, r_count, dtype, vdest, 0,
                 tmp_send, s_count, dtype, vdest, 0,
                 comm, MPI_STATUS_IGNORE);
    
    w_size <<= 1;
  }

  free(tmp_buf_raw);

  return MPI_SUCCESS;
}
