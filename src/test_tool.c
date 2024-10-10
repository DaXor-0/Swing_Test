#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "test_tool.h"

// Write on random array of lenght len
void rand_array_generator(int *target, size_t len, int rank){
  unsigned int seed = time(NULL) + rank; 
  
  for (size_t i = 0; i < len; i++){
    target[i] = (int) rand_r(&seed) % 1000;
  }
}

// Check if buffers are equal
int are_equal(const void *buf_1, const void *buf_2, size_t len){
  if (len == 0) return 1;

  char *b1 = (char *) buf_1;
  char *b2 = (char *) buf_2;

  size_t i;
  for (i = 0; i < len; i++){
    if (b1[i] != b2[i]){
      return 0;
    }
  }
  return 1;
}

// Get algorithm number from file "collective_rules.txt" to format output file
int get_alg_number(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Could not open file %s\n", filename);
    return -1;
  }

  char line[256];
  int line_number = 1;
  int alg_number = 0;

  while (fgets(line, sizeof(line), file)) {
    if (line_number == 6) {
      if (sscanf(line, "%*d %d", &alg_number) != 1) {
        fprintf(stderr, "Failed to read algorithm number from line 6.\n");
        fclose(file);
        return -1;  // Return -1 if reading fails
      }
    break;
    }
  line_number++;
  }

  fclose(file);
  return alg_number;  // Return the read algorithm number
}


int create_filename(char *filename, size_t fn_size, int comm_sz, size_t array_size, int valid){
  // Get the current time
  time_t now = time(NULL);
  struct tm *tm_info = localtime(&now);
  char time_string[20];  // Enough space for a formatted date string like "YYYY-MM-DD_HH-MM-SS"
  
  // Format the time string
  strftime(time_string, sizeof(time_string), "%Y-%m-%d_%H-%M-%S", tm_info);
  
  char *dynamic_rules = getenv("OMPI_MCA_coll_tuned_use_dynamic_rules");
  if (dynamic_rules == NULL){
    fprintf(stderr, "Failed to retrieve dynamic_rules env var.\n");
    return -1;
  }

  // If I'm using a custom algorithm, the filename starts with the number of the algorithm, otherwise 0
  if (strcmp(dynamic_rules, "1") == 0){
    // Construct the output filename
    int alg_number = get_alg_number("./collective_rules.txt");
    if (alg_number == -1) {
      fprintf(stderr, "Failed to retrieve algorithm number.\n");
      return -1; // Exit if the algorithm number could not be retrieved
    }
    
    snprintf(filename, fn_size, (valid == 1) ? "%d_%d_%ld___%s___true.txt" : "%d_%d_%ld___%s___false.txt", alg_number, comm_sz, array_size, time_string);
  }
  else{
    snprintf(filename, fn_size, (valid == 1) ? "0_%d_%ld___%s___true.txt" : "0_%d_%ld___%s___false.txt", comm_sz, array_size, time_string);
  }
  return 0;
}


int concatenate_path(const char *dirpath, const char *filename, char *fullpath){
  if (dirpath == NULL || filename == NULL) {
    fprintf(stderr, "Error: Directory path or filename is NULL.\n");
    return -1;
  }

  // Check if the lengths of dirpath and filename are within the allowed size
  size_t dirpath_len = strlen(dirpath);
  size_t filename_len = strlen(filename);

  if (dirpath_len == 0) {
    fprintf(stderr, "Error: Directory path is empty.\n");
    return -1;
  }

  // Ensure the final full path won't exceed buffer size
  if (dirpath_len + filename_len + 2 > MAX_PATH_LENGTH) {
    // +2 accounts for a possible '/' and the null terminator
    fprintf(stderr, "Error: Combined path length exceeds buffer size.\n");
    return -1;
  }

  // Initialize fullpath with the directory path
  strcpy(fullpath, dirpath);

  // Concatenate the filename to the directory path
  strcat(fullpath, filename);

  return 0;
}
