#ifndef TEST_TOOLS_H
#define TEST_TOOLS_H

#include <mpi.h>
#include <stddef.h>

#include "test_selection.h"

#define TEST_MAX_PATH_LENGTH 512
#define TEST_BASE_EPSILON_FLOAT 1e-6    // Base epsilon for float
#define TEST_BASE_EPSILON_DOUBLE 1e-15  // Base epsilon for double


int get_data_type(const char *type_string, MPI_Datatype *dtype, size_t *type_size);

int rand_array_generator(void *target, const char *type_string, size_t array_size, int rank);

int are_equal_eps(const void *buf_1, const void *buf_2, size_t array_size,
                  const char *type_string, int comm_sz);

int concatenate_path(const char *dirpath, const char *filename, char *fullpath);

int get_command_line_arguments(int argc, char** argv, size_t *array_size, int* iter,
                               const char **type_string, allreduce_algo_t *algorithm,
                               const char ** outputdir);

int write_output_to_file(const char *fullpath, double *highest, double *all_times, int iter);

int write_allocations_to_file(const char* filename, MPI_Comm comm);

int file_not_exists(const char* filename);


// typedef enum{
//   ALLREDUCE,
//   ALLGATHER,
//   REDUCE_SCATTER,
//   BROADCAST,
//   REDUCE
// } coll_t;
//
// typedef enum{
//   OMPI,
//   MPICH,
//   CUSTOM_OMPI
// } mpi_impl_t;
//
//
// typedef struct {
//   mpi_impl_t implementation;
//   coll_t collective;
// } target;

#endif