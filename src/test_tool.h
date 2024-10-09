#ifndef TEST_TOOLS_H
#define TEST_TOOLS_H

#include <stdlib.h>

void rand_array_generator(int *target, size_t len, int rank);
int are_equal(const void *buf_1, const void *buf_2, size_t len);
int get_alg_number(const char *filename);
int create_filename(char *filename, size_t fn_size, int comm_sz, size_t array_size);

#endif
