#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <sys/stat.h>

#include "bench_utils.h"

/**
 * @brief Converts a string to a `coll_t` enum value.
 *
 * @param coll_str String representing the collective type (e.g., "ALLREDUCE").
 * @return A `coll_t` enum value corresponding to the input string.
 *         Returns `COLL_UNKNOWN` for invalid strings.
 */
static inline coll_t get_collective_from_string(const char *coll_str) {
  CHECK_STR(coll_str, "ALLREDUCE", ALLREDUCE);
  CHECK_STR(coll_str, "ALLGATHER", ALLGATHER);
  CHECK_STR(coll_str, "BCAST", BCAST);
  CHECK_STR(coll_str, "REDUCE_SCATTER", REDUCE_SCATTER);
  return COLL_UNKNOWN;
}

/**
* @brief Select and returns the appropriate allocator function based
* on the collective type. It returns NULL if the collective type is 
* not supported.
*
* @param collectove `coll_t` enum value representing the collective type.
*
* @return Pointer to the selected allocator function, or NULL if the
*         collective type is not supported.
*/
static inline allocator_func_ptr get_allocator(coll_t collective) {
  switch (collective) {
    case ALLREDUCE:
      return allreduce_allocator;
    case ALLGATHER:
      return allgather_allocator;
    case BCAST:
      return bcast_allocator;
    case REDUCE_SCATTER:
      return reduce_scatter_allocator;
    default:
      return NULL;
  }
}

/**
* @brief Select and returns the appropriate allreduce function based
* on the algorithm. It returns the default allreduce function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal allreduce function.
*/
static inline allreduce_func_ptr get_allreduce_function(const char *algorithm) {
  CHECK_STR(algorithm, "recursive_doubling_over", allreduce_recursivedoubling);
  CHECK_STR(algorithm, "ring_over", allreduce_ring);
  CHECK_STR(algorithm, "rabenseifner_over", allreduce_rabenseifner);
  CHECK_STR(algorithm, "swing_lat_over", allreduce_swing_lat);
  CHECK_STR(algorithm, "swing_bdw_static_over", allreduce_swing_bdw_static);
  CHECK_STR(algorithm, "swing_bdw_remap_over", allreduce_swing_bdw_remap);

  BENCH_DEBUG_PRINT_STR("MPI_Allreduce");
  return allreduce_wrapper;
}

/**
* @brief Select and returns the appropriate allgather function based
* on the algorithm. It returns the default allgather function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal allgather function.
*/
static inline allgather_func_ptr get_allgather_function(const char *algorithm) {
  // if(strcmp(algorithm, "K_BRUCK_OVER") == 0 ) return allgather_k_bruck;
  CHECK_STR(algorithm, "recursive_doubling_over", allgather_recursivedoubling);
  CHECK_STR(algorithm, "recursive_distance_doubling_over", allgather_recursive_distance_doubling);
  CHECK_STR(algorithm, "ring_over", allgather_ring);
  CHECK_STR(algorithm, "swing_static_memcpy_over", allgather_swing_static_memcpy);
  CHECK_STR(algorithm, "swing_static_send_over", allgather_swing_static_send);

  BENCH_DEBUG_PRINT_STR("MPI_Allgather");
  return allgather_wrapper;
}

/**
* @brief Select and returns the appropriate bcast function based
* on the algorithm. It returns the default bcast function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal bcast function.
*/
static inline bcast_func_ptr get_bcast_function(const char *algorithm) {
  CHECK_STR(algorithm, "scatter_allgather_over", bcast_scatter_allgather);
  CHECK_STR(algorithm, "swing_lat_over", bcast_swing_lat);
  CHECK_STR(algorithm, "swing_lat_reversed_over", bcast_swing_lat_reversed);
  CHECK_STR(algorithm, "swing_bdw_static_over", bcast_swing_bdw_static);
//  CHECK_STR(algorithm, "swing_bdw_static_reversed_over", bcast_swing_bdw_static_reversed);

  BENCH_DEBUG_PRINT_STR("MPI_Bcast");
  return bcast_wrapper;
}


/**
* @breif Select and returns the appropriate reduce scatter function based
* on the algorithm. It returns the default reduce scatter function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal reduce scatter function.
*/
static inline reduce_scatter_func_ptr get_reduce_scatter_function (const char *algorithm){
  CHECK_STR(algorithm, "recursive_halving_over", reduce_scatter_recursivehalving);
  CHECK_STR(algorithm, "ring_over", reduce_scatter_ring);
  CHECK_STR(algorithm, "butterfly_over", reduce_scatter_butterfly);
  CHECK_STR(algorithm, "swing_static_over", reduce_scatter_swing_static);

  BENCH_DEBUG_PRINT_STR("MPI_Reduce_scatter");
  return MPI_Reduce_scatter;
}


int get_routine(test_routine_t *test_routine, const char *algorithm) {
  const char *coll_str = NULL;

  // Get the collective type from the environment variable
  coll_str = getenv("COLLECTIVE_TYPE");
  if(NULL == coll_str) {
    fprintf(stderr, "Error! `COLLECTIVE_TYPE` environment \
                    variable not set. Aborting...");
    return -1;
  }

  // Convert the collective string to a `coll_t` enum value
  test_routine->collective = get_collective_from_string(coll_str);
  if(test_routine->collective == COLL_UNKNOWN) {
    fprintf(stderr, "Error! Invalid `COLLECTIVE_TYPE` value: \
                     %s. Aborting...", coll_str);
    return -1;
  }

  // Set the right allocator based on the collective type
  test_routine->allocator = get_allocator(test_routine->collective);
  if(NULL == test_routine->allocator) {
    fprintf(stderr, "Error! Allocator is NULL. Aborting...");
    return -1;
  }

  // Set the right function pointer based on the collective type and algorithm
  switch (test_routine->collective){
    case ALLREDUCE:
      test_routine->function.allreduce = get_allreduce_function(algorithm);
      break;
    case ALLGATHER:
      test_routine->function.allgather = get_allgather_function(algorithm);
      break;
    case BCAST:
      test_routine->function.bcast = get_bcast_function(algorithm);
      break;
    case REDUCE_SCATTER:
      test_routine->function.reduce_scatter = get_reduce_scatter_function(algorithm);
      break;
    default :
      fprintf(stderr, "Error! Invalid collective type. Aborting...");
      return -1;
    }

  return 0;
}



int test_loop(test_routine_t test_routine, void *sbuf, void *rbuf, size_t count,
              MPI_Datatype dtype, MPI_Comm comm, int iter, double *times){
  int rank, comm_sz, ret, *rcounts = NULL;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  
  switch (test_routine.collective){
    case ALLREDUCE:
      ret = allreduce_test_loop(sbuf, rbuf, count, dtype, MPI_SUM, comm, iter,
                          times, test_routine);
      break;
    case ALLGATHER:
      ret = allgather_test_loop(sbuf, count / (size_t) comm_sz, dtype,
                          rbuf, count / (size_t) comm_sz, dtype,
                          comm, iter, times, test_routine);
      break;
    case BCAST:
      ret = bcast_test_loop(sbuf, count, dtype, 0, comm, iter, times,
                      test_routine);
      break;
    case REDUCE_SCATTER:
      rcounts = (int *)malloc(comm_sz * sizeof(int));
      for(int i = 0; i < comm_sz; i++) { rcounts[i] = count / comm_sz; }
      ret = reduce_scatter_test_loop(sbuf, rbuf, rcounts, dtype, MPI_SUM, comm, iter,
                               times, test_routine);
      free(rcounts);
      break;
    default:
      fprintf(stderr, "still not implemented, aborting...");
      return -1;
  }
  return ret;
}


int ground_truth_check(test_routine_t test_routine, void *sbuf, void *rbuf,
                       void *rbuf_gt, size_t count, MPI_Datatype dtype, MPI_Comm comm){
  int rank, comm_sz, *rcounts = NULL, ret = 0, type_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  MPI_Type_size(dtype, &type_size);

  switch (test_routine.collective){
    case ALLREDUCE:
      PMPI_Allreduce(sbuf, rbuf_gt, count, dtype, MPI_SUM, comm);
      GT_CHECK_BUFFER(rbuf, rbuf_gt, count, dtype, comm);
      break;
    case ALLGATHER:
      PMPI_Allgather(sbuf, count / (size_t) comm_sz, dtype, \
                 rbuf_gt, count / (size_t) comm_sz, dtype, comm);
      GT_CHECK_BUFFER(rbuf, rbuf_gt, count, dtype, comm);
      break;
    case BCAST:
      if(rank == 0) {
        memcpy(rbuf_gt, sbuf, count * type_size);
      }
      PMPI_Bcast(rbuf_gt, count, dtype, 0, comm);
      GT_CHECK_BUFFER(sbuf, rbuf_gt, count, dtype, comm);
      break;
    case REDUCE_SCATTER:
      rcounts = (int *)malloc(comm_sz * sizeof(int));
      for(int i = 0; i < comm_sz; i++){
        rcounts[i] = count / comm_sz;
      }
      PMPI_Reduce_scatter(sbuf, rbuf_gt, rcounts, dtype, MPI_SUM, comm);
      GT_CHECK_BUFFER(rbuf, rbuf_gt, rcounts[rank], dtype, comm);
      free(rcounts);
      break;
    default:
      fprintf(stderr, "still not implemented, aborting...");
      return -1;
  }
  return ret;
}


int get_command_line_arguments(int argc, char** argv, size_t *array_count, int* iter,
                               const char **algorithm, const char **type_string) {
  if(argc != 5) {
    fprintf(stderr, "Usage: %s <array_count> <iterations> <algorithm> <dtype>", argv[0]);
    return -1;
  }

  char *endptr;
  *array_count = (size_t) strtoll(argv[1], &endptr, 10);
  if(*endptr != '\0' || *array_count <= 0) {
    fprintf(stderr, "Error: Invalid array count. It must be a positive integer. Aborting...");
    return -1;
  }

  *iter = (int) strtol(argv[2], &endptr, 10);
  if(*endptr != '\0' || *iter <= 0) {
    fprintf(stderr, "Error: Invalid number of iterations. It must be a positive integer. Aborting...");
    return -1;
  }

  *algorithm = argv[3];

  *type_string = argv[4];

  return 0;
}


/**
 * @struct TypeMap
 * @brief Maps type names to corresponding MPI data types and sizes.
 */
typedef struct {
  const char* t_string;   /**< Type name as a string. */
  MPI_Datatype mpi_type;  /**< Corresponding MPI datatype. */
  size_t t_size;          /**< Size of the datatype in bytes. */
} TypeMap;

/**
 * @brief Static array mapping string representations to MPI datatypes. Will be
 *        used to map command-line input argument to datatype and its size.
 */
const static TypeMap type_map[] = {
  {"int8",          MPI_INT8_T,         sizeof(int8_t)},
  {"int16",         MPI_INT16_T,        sizeof(int16_t)},
  {"int32",         MPI_INT32_T,        sizeof(int32_t)},
  {"int64",         MPI_INT64_T,        sizeof(int64_t)},
  {"int",           MPI_INT,            sizeof(int)},
  {"float",         MPI_FLOAT,          sizeof(float)},
  {"double",        MPI_DOUBLE,         sizeof(double)},
  {"char",          MPI_CHAR,           sizeof(char)},
  {"unsigned_char", MPI_UNSIGNED_CHAR,  sizeof(unsigned char)}
};


int get_data_type(const char *type_string, MPI_Datatype *dtype, size_t *type_size) {
  int num_types = sizeof(type_map) / sizeof(type_map[0]);

  for(int i = 0; i < num_types; i++) {
    if(strcmp(type_string, type_map[i].t_string) == 0) {
      *dtype = type_map[i].mpi_type;
      *type_size = type_map[i].t_size;
      return 0;
    }
  }

  fprintf(stderr, "Error: datatype %s not in `type_map`. Aborting...", type_string);
  return -1;
}

/**
* @breif Writes all timing results to a specified output file in CSV format.
*
* @param fullpath  The full path to the output file.
* @param highest   An array containing the highest timing values for each iteration.
* @param all_times A 2D array flattened into 1D containing timing values for all ranks
*                  across all iterations.
* @param iter      The number of iterations.
*
* @return int Returns 0 on success, or -1 if an error occurs.
* @note Time is saved in ns (i.e. 10^-9 s).
*/
static inline int write_all_output_to_file(const char *fullpath, double *highest, double *all_times, int iter) {
  int comm_sz;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  FILE *output_file = fopen(fullpath, "w");
  if(output_file == NULL) {
    fprintf(stderr, "Error: Opening file %s for writing", fullpath);
    return -1;
  }

  // Write the header with ranks from rank0 to rankN
  fprintf(output_file, "highest");
  for(int rank = 0; rank < comm_sz; rank++) {
    fprintf(output_file, ",rank%d", rank);
  }
  fprintf(output_file, "\n");

  // Write the timing data
  for(int i = 0; i < iter; i++) {
    fprintf(output_file, "%" PRId64, (int64_t)(highest[i] * 1e9));
    for(int j = 0; j < comm_sz; j++) {
      fprintf(output_file, ",%" PRId64, (int64_t)(all_times[j * iter + i] * 1e9));
    }
    fprintf(output_file, "\n");
  }

  fclose(output_file);
  return 0;
}

/**
* @brief Writes the summarized timing results to a specified output file in CSV format.
*
* @param fullpath The full path to the output file.
* @param highest An array containing the highest timing values for each iteration.
* @param iter The number of iterations.
*
* @return int Returns 0 on success, or -1 if an error occurs.
*
* @note Time is saved in ns (i.e. 10^-9 s).
*/
static inline int write_summarized_output_to_file(const char *fullpath, double *highest, int iter){
  FILE *output_file = fopen(fullpath, "w");
  if(output_file == NULL) {
    fprintf(stderr, "Error: Opening file %s for writing", fullpath);
    return -1;
  }

  // Write the header with ranks from rank0 to rankN
  fprintf(output_file, "highest\n");

  // Write the timing data
  for(int i = 0; i < iter; i++) {
    fprintf(output_file, "%" PRId64"\n", (int64_t)(highest[i] * 1e9));
  }

  fclose(output_file);
  return 0;
}

int write_output_to_file(const char *output_level, const char *filename, double *highest, double *all_times, int iter){
  if(strcmp(output_level, "all") == 0) {
    return write_all_output_to_file(filename, highest, all_times, iter);
  } else if(strcmp(output_level, "summarized") == 0) {
    return write_summarized_output_to_file(filename, highest, iter);
  } else {
    fprintf(stderr, "Error: Output level %s not recognized. Aborting...", output_level);
    return -1;
  }
}

int file_not_exists(const char* filename) {
  struct stat buffer;
  return (stat(filename, &buffer) != 0) ? 1 : 0;
}


int write_allocations_to_file(const char* filename, MPI_Comm comm) {
  int rank, comm_sz;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  MPI_Get_processor_name(processor_name, &name_len);

  MPI_File file;
  if(MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
    if(rank == 0) {
      fprintf(stderr, "Error: Opening file %s for writing", filename);
    }
    return MPI_ERR_FILE;
  }

  const char header[] = "MPI_Rank,allocation\n";
  // Rank 0 writes the header to the file
  if(rank == 0) {
    MPI_File_write_at(file, 0, header, sizeof(header) - 1, MPI_CHAR, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(comm);  // Ensure header is written before writing rank data

  // Define a fixed-length buffer for each rank's entry
  char buffer[MPI_MAX_PROCESSOR_NAME + 16];  // Fixed space for rank, comma, name, and newline
  snprintf(buffer, sizeof(buffer), "%d,%s\n", rank, processor_name);

  // Calculate a unique offset for each rank using fixed-size entries
  MPI_Offset offset = sizeof(header) - 1 + rank * (MPI_MAX_PROCESSOR_NAME + 16);

  // Write each rank's data at its calculated offset
  MPI_File_write_at(file, offset, buffer, strlen(buffer), MPI_CHAR, MPI_STATUS_IGNORE);

  MPI_File_close(&file);
  return MPI_SUCCESS;
}


int rand_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t count,
                         MPI_Comm comm, test_routine_t test_routine) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  unsigned int seed = time(NULL) + rank;
  
  // For BCAST, only rank 0 generates the sendbuf
  if(test_routine.collective == BCAST && rank != 0) {
    return 0;
  }
  // If generating sendbuf for an ALLGATHER, the number of element is
  // count / comm_sz
  size_t real_sbuf_count =
    (test_routine.collective == ALLGATHER) ?
                                        count / (size_t) comm_sz : count;

  for(size_t i = 0; i < real_sbuf_count; i++) {
    if(dtype == MPI_INT8_T) {
      ((int8_t *)sbuf)[i] = (int8_t)((rand_r(&seed) % 256) - 128);
    } else if(dtype == MPI_INT16_T) {
      ((int16_t *)sbuf)[i] = (int16_t)((rand_r(&seed) % 65536) - 32768);
    } else if(dtype == MPI_INT32_T) {
      ((int32_t *)sbuf)[i] = (int32_t)(rand_r(&seed));
    } else if(dtype == MPI_INT64_T) {
      ((int64_t *)sbuf)[i] = ((int64_t)rand_r(&seed) << 32) | rand_r(&seed);
    } else if(dtype == MPI_INT) {
      ((int *)sbuf)[i] = (int)rand_r(&seed);
    } else if(dtype == MPI_FLOAT) {
      ((float *)sbuf)[i] = (float)rand_r(&seed) / (float) RAND_MAX * 100.0f;
    } else if(dtype == MPI_DOUBLE) {
      ((double *)sbuf)[i] = (double)rand_r(&seed) / (double) RAND_MAX * 100.0;
    } else if(dtype == MPI_CHAR) {
      ((char *)sbuf)[i] = (char)((rand_r(&seed) % 256) - 128);
    } else if(dtype == MPI_UNSIGNED_CHAR) {
      ((unsigned char *)sbuf)[i] = (unsigned char)(rand_r(&seed) % 256);
    } else {
      fprintf(stderr, "Error: sbuf not generated correctly. Aborting...");
      return -1;
    }
  }

  return 0;
}


int concatenate_path(const char *dir_path, const char *filename, char *fullpath) {
  if(dir_path == NULL || filename == NULL) {
    fprintf(stderr, "Directory path or filename is NULL.");
    return -1;
  }

  size_t dir_path_len = strlen(dir_path);
  size_t filename_len = strlen(filename);

  if(dir_path_len == 0) {
    fprintf(stderr, "Directory path is empty.");
    return -1;
  }

  if(dir_path_len + filename_len + 2 > BENCH_MAX_PATH_LENGTH) {
    fprintf(stderr, "Combined path length exceeds buffer size.");
    return -1;
  }

  strcpy(fullpath, dir_path);
  if(dir_path[dir_path_len - 1] != '/') {
    strcat(fullpath, "/");
  }
  strcat(fullpath, filename);

  return 0;
}


int are_equal_eps(const void *buf_1, const void *buf_2, size_t count,
                  MPI_Datatype dtype, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  if(count == 0) return 0;

  if(MPI_FLOAT == dtype) {
    float *b1 = (float *) buf_1;
    float *b2 = (float *) buf_2;

    float epsilon = comm_sz * BENCH_BASE_EPSILON_FLOAT * 100.0f;

    for(size_t i = 0; i < count; i++) {
      if(fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  } else if(MPI_DOUBLE == dtype) {
    double *b1 = (double *) buf_1;
    double *b2 = (double *) buf_2;

    double epsilon = comm_sz * BENCH_BASE_EPSILON_DOUBLE * 100.0;

    for(size_t i = 0; i < count; i++) {
      if(fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  }

  return 0;
}

#ifdef DEBUG
/**
 * @brief Little helper function to calculate integer powers for debugging purposes.
 *
 * @return base^power as an int64_t.
 */
static inline int64_t int_pow_64(int base, int exp) {
  int64_t result = 1, base_ = base;
  if(exp == 0) return 1;

  while (exp > 0) {
    if(exp % 2 == 1) result *= base_;
    base_ *= base_;
    exp /= 2;
  }
  return result;
}

/**
 * @brief Little helper function to calculate integer powers for debugging purposes.
 *
 * @return base^power as an int32_t.
 */
static inline int32_t int_pow_32(int base, int exp) {
  int32_t result = 1, base_ = base;
  if(exp == 0) return 1;

  while (exp > 0) {
    if(exp % 2 == 1) result *= base_;
    base_ *= base_;
    exp /= 2;
  }
  return result;
}

/**
 * @brief Little helper function to calculate integer powers for debugging purposes.
 *
 * @return base^power as an int.
 */
static inline int int_pow(int base, int exp) {
  int result = 1;
  if(exp == 0) return 1;

  while (exp > 0) {
    if(exp % 2 == 1) result *= base;
    base *= base;
    exp /= 2;
  }
  return result;
}

int debug_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t count,
                         MPI_Comm comm, test_routine_t test_routine) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  // For BCAST, only rank 0 has a valid sbuf
  if(test_routine.collective == BCAST && rank != 0) {
    return 0;
  }

  size_t real_sbuf_count =
    (test_routine.collective == ALLGATHER) ?
                                        count / (size_t) comm_sz : count;

  for(int i=0; i< real_sbuf_count; i++){
    if(dtype == MPI_INT64_T) {
      ((int64_t*)sbuf)[i] = int_pow_64(10, rank);
    } else if(dtype == MPI_INT32_T) {
      ((int32_t*)sbuf)[i] = int_pow_32(10, rank);
    } else if(dtype == MPI_INT){
      ((int*)sbuf)[i] = int_pow(10, rank);
    } else {
      fprintf(stderr, "Error: Datatype not implemented for `debug_sbuf_init`...");
      return -1;
    }
  }
  return 0;
}

/**
 * Helper function to print an array of elements based on MPI_Datatype.
 */
static void print_buffer_helper(const void *buf, size_t count, MPI_Datatype dtype) {
  if(dtype == MPI_INT64_T) {
    const int64_t *data = (const int64_t *)buf;
    for(size_t j = 0; j < count; j++) {
      printf("%" PRId64 " ", data[j]);
    }
  } else if(dtype == MPI_INT32_T) {
    const int32_t *data = (const int32_t *)buf;
    for(size_t j = 0; j < count; j++) {
      printf("%" PRId32 " ", data[j]);
    }
  } else if(dtype == MPI_INT) {
    const int *data = (const int *)buf;
    for(size_t j = 0; j < count; j++) {
      printf("%d ", data[j]);
    }
  } else {
    fprintf(stderr, "Error: Datatype print not supported.");
  }
}

void debug_print_buffers(const void *rbuf, const void *rbuf_gt, size_t count,
                        MPI_Datatype dtype, MPI_Comm comm, int use_barrier) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  for(int i = 0; i < comm_sz; i++) {
    if(rank == i) {
      printf("\nRank %d:\nrecvbuf: ", rank);
      print_buffer_helper(rbuf, count, dtype);
      printf("\ng_truth: ");
      print_buffer_helper(rbuf_gt, count, dtype);
      printf("\n\n");
    }
    fflush(stdout);
    if(use_barrier == 0) MPI_Barrier(comm);
  }
}

#endif // DEBUG
