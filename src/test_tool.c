#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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


int create_filename(char *filename, size_t fn_size, int comm_sz, size_t array_size){
  // Get the current time
  time_t now = time(NULL);
  struct tm *tm_info = localtime(&now);
  char time_string[20];  // Enough space for a formatted date string like "YYYY-MM-DD_HH-MM-SS"
  
  // Format the time string
  strftime(time_string, sizeof(time_string), "%Y-%m-%d_%H-%M-%S", tm_info);
  
  // Construct the output filename
  int alg_number = get_alg_number("./collective_rules.txt");
  if (alg_number == -1) {
    fprintf(stderr, "Failed to retrieve algorithm number.\n");
    return -1; // Exit if the algorithm number could not be retrieved
  }
  snprintf(filename, fn_size, "%s_%d_%ld_%d.txt", time_string, comm_sz, array_size, alg_number);

  return 0;
}


