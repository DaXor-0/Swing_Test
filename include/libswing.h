#ifndef LIBSWING_H
#define LIBSWING_H

#include <mpi.h>
#include <stddef.h>

int allreduce_swing_lat(const void *sbuf, void *rbuf, size_t count,
                                MPI_Datatype dtype, MPI_Op op, MPI_Comm comm);

int allreduce_swing_bdw_static(const void *send_buf, void *recv_buf, size_t count,
                                MPI_Datatype dtype, MPI_Op op, MPI_Comm comm);

int allreduce_recursivedoubling(const void *sbuf, void *rbuf, size_t count,
                                MPI_Datatype dtype, MPI_Op op, MPI_Comm comm);

#endif