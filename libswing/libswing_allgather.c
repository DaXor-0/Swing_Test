#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "libswing.h"
#include "libswing_utils.h"

int allgather_rabenseifner(const void *sbuf, size_t scount, MPI_Datatype sdtype,
                           void* rbuf, size_t rcount, MPI_Datatype rdtype, MPI_Comm comm)
{
    int line = -1, rank, size, pow2size, err;
    int remote, distance, sendblocklocation;
    ptrdiff_t rlb, rext;
    char *tmpsend = NULL, *tmprecv = NULL;

    err = MPI_Comm_size(comm, &size);
    err = MPI_Comm_rank(comm, &rank);

    pow2size = next_poweroftwo (size);
    pow2size >>=1;

    /* Current implementation only handles power-of-two number of processes.
       If the function was called on non-power-of-two number of processes,
       print warning and call bruck allgather algorithm with same parameters.
    */
    if (pow2size != size) {
        fprintf(stderr, "ERROR! Rabenseifner allgather works only with po2 ranks!\n");
        goto err_hndl;

    }

    err = MPI_Type_get_extent (rdtype, &rlb, &rext);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    /* Initialization step:
       - if send buffer is not MPI_IN_PLACE, copy send buffer to block 0 of
       receive buffer
    */
    if (MPI_IN_PLACE != sbuf) {
        tmpsend = (char*) sbuf;
        tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
        err = copy_buffer_different_dt(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }

    }

    /* Communication step:
       At every step i, rank r:
       - exchanges message with rank remote = (r ^ 2^i).

    */
    sendblocklocation = rank;
    for (distance = 0x1; distance < size; distance<<=1) {
        remote = rank ^ distance;

        if (rank < remote) {
            tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
            tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation + distance) * (ptrdiff_t)rcount * rext;
        } else {
            tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
            tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation - distance) * (ptrdiff_t)rcount * rext;
            sendblocklocation -= distance;
        }

        /* Sendreceive */
        err = MPI_Sendrecv(tmpsend, (ptrdiff_t)distance * (ptrdiff_t)rcount, rdtype, remote, 0,
                                       tmprecv, (ptrdiff_t)distance * (ptrdiff_t)rcount, rdtype,
                                       remote, 0, comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    }

    return MPI_SUCCESS;

 err_hndl:
    fprintf(stderr, "%s:%4d\tRank %d Error occurred %d\n", __FILE__, line, rank, err);
    (void)line;  // silence compiler warning
    return err;
}