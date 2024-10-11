#ifndef TEST_TOOLS_H
#define TEST_TOOLS_H

#include <stdlib.h>
#include <mpi.h>

#define MAX_PATH_LENGTH 512

typedef struct {
  const char* t_string;
  MPI_Datatype mpi_type;
  size_t t_size;
} TypeMap;

extern const TypeMap type_map[];

int get_data_type(const char *type_string, MPI_Datatype *dtype, size_t *type_size);

int rand_array_generator(void *target, const char *type_string, size_t array_size, int rank);

int are_equal(const void *buf_1, const void *buf_2, size_t len);

int get_alg_number(const char *filename, int *alg_number);

int create_filename(char *filename, size_t fn_size, int comm_sz, size_t array_size, const char *type_string);

int concatenate_path(const char *dirpath, const char *filename, char *fullpath);

#endif
