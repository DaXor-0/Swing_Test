#include <stdio.h>
#include <string.h>
#include <time.h>

#include "test_tool.h"

const TypeMap type_map[] = {
  {"int8", MPI_INT8_T, sizeof(int8_t)},
  {"int16", MPI_INT16_T, sizeof(int16_t)},
  {"int32", MPI_INT32_T, sizeof(int32_t)},
  {"int64", MPI_INT64_T, sizeof(int64_t)},
  {"int", MPI_INT, sizeof(int)},
  {"float", MPI_FLOAT, sizeof(float)},
  {"double", MPI_DOUBLE, sizeof(double)},
  {"char", MPI_CHAR, sizeof(char)}
};


int get_data_type(const char *type_string, MPI_Datatype *dtype, size_t *type_size) {
  int num_types = sizeof(type_map) / sizeof(TypeMap);
  
  for (int i = 0; i < num_types; i++) {
    if (strcmp(type_string, type_map[i].t_string) == 0) {
      *dtype = type_map[i].mpi_type;
      *type_size = type_map[i].t_size;
      return 0;
    }
  }

  return -1;
}


// Write on random array of length len
int rand_array_generator(void *target, const char *type_string, size_t array_size, int rank){
  unsigned int seed = time(NULL) + rank; 
  // Compare the type string to determine the type and fill the array
  if (strcmp(type_string, "int8") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int8_t *)target)[i] = (int8_t)(rand_r(&seed) % 256) - 128; // Range from -128 to 127
    }
  } else if (strcmp(type_string, "int16") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int16_t *)target)[i] = (int16_t)(rand_r(&seed) % 65536) - 32768; // Range from -32768 to 32767
    }
  } else if (strcmp(type_string, "int32") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int32_t *)target)[i] = (int32_t)(rand_r(&seed)); // Use full int32_t range
    }
  } else if (strcmp(type_string, "int64") == 0) {
      for (size_t i = 0; i < array_size; i++) {
      ((int64_t *)target)[i] = ((int64_t)rand_r(&seed) << 32) | rand_r(&seed); // Create a 64-bit value
    }
  } else if (strcmp(type_string, "int") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int *)target)[i] = (int)rand_r(&seed); // Random int between 0 and 99
    }
  } else if (strcmp(type_string, "float") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((float *)target)[i] = (float)rand_r(&seed) / RAND_MAX * 100.0f; // Random float between 0.0 and 100.0
    }
  } else if (strcmp(type_string, "double") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((double *)target)[i] = (double)rand_r(&seed) / RAND_MAX * 100.0; // Random double between 0.0 and 100.0
    }
  } else if (strcmp(type_string, "char") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((char *)target)[i] = (char)(rand_r(&seed) % 256); // Random char value
    }
  } else {
    return -1;  // Unsupported type
  }

  return 0;  // Success
}


// Check if buffers are equal up to len
// NOTE: LEN IS IN NUMBER OF BYTES
int are_equal(const void *buf_1, const void *buf_2, size_t len){
  if (len == 0) return 0;

  char *b1 = (char *) buf_1;
  char *b2 = (char *) buf_2;

  size_t i;
  for (i = 0; i < len; i++){
    if (b1[i] != b2[i]){
      return -1;
    }
  }
  return 0;
}

// Get algorithm number from file "collective_rules.txt" to format output file
int get_alg_number(const char *filename, int *alg_number) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Failed to open file %s\n", filename);
    return -1;
  }

  char line[256];
  int line_number = 1;

  while (fgets(line, sizeof(line), file)) {
    if (line_number == 6) {
      if (sscanf(line, "%*d %d", alg_number) != 1) {
        fprintf(stderr, "Failed to read algorithm number from line 6.\n");
        fclose(file);
        return -1;  // Return -1 if reading fails
      }
    break;
    }
  line_number++;
  }

  fclose(file);

  return 0;
}


int create_filename(char *filename, size_t fn_size, int comm_sz, size_t array_size, const char *type_string, const char *rulepath){
  char *dynamic_rules = getenv("OMPI_MCA_coll_tuned_use_dynamic_rules");
  if (dynamic_rules == NULL){
    fprintf(stderr, "Failed to retrieve dynamic_rules env var.\n");
    return -1;
  }

  // If I'm selectiong an algorithm, the filename starts with the number of the algorithm, otherwise 0
  if (strcmp(dynamic_rules, "1") == 0){
    int alg_number;
    if (get_alg_number(rulepath, &alg_number) == -1){
      fprintf(stderr, "Failed to retrieve algorithm number.\n");
      return -1;
    }
    
    snprintf(filename, fn_size, "%d_%ld_%s_%d.csv", comm_sz, array_size, type_string, alg_number);
  }
  else{
    snprintf(filename, fn_size, "%d_%ld_%s_0.csv", comm_sz, array_size, type_string);
  }
  return 0;
}


int concatenate_path(const char *dirpath, const char *filename, char *fullpath){
  if (dirpath == NULL || filename == NULL) {
    fprintf(stderr, "Directory path or filename is NULL.\n");
    return -1;
  }

  // Check if the lengths of dirpath and filename are within the allowed size
  size_t dirpath_len = strlen(dirpath);
  size_t filename_len = strlen(filename);

  if (dirpath_len == 0) {
    fprintf(stderr, "Directory path is empty.\n");
    return -1;
  }

  // Ensure the final full path won't exceed buffer size
  if (dirpath_len + filename_len + 2 > MAX_PATH_LENGTH) {
    // +2 accounts for a possible '/' and the null terminator
    fprintf(stderr, "Combined path length exceeds buffer size.\n");
    return -1;
  }

  // Initialize fullpath with the directory path
  strcpy(fullpath, dirpath);

  // Concatenate the filename to the directory path
  strcat(fullpath, filename);

  return 0;
}
