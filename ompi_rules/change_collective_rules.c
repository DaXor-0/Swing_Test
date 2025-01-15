#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 256

int update_file(const char *filename, int new_value) {
  // Validate the new value
  if (new_value < 0 || new_value > 16) {
    fprintf(stderr, "Error: The number must be between 0 and 16.\n");
    return -1;
  }

  if (new_value > 13) new_value = 0;
  

  FILE *file = fopen(filename, "r+");
  if (file == NULL) {
    fprintf(stderr, "Error opening the file.\n");
    return -1;
  }

  char line[MAX_LINE_LENGTH];
  int line_number = 0;
  long sixth_line_position = 0;

  // Find the position of the sixth line
  while (fgets(line, sizeof(line), file)) {
    line_number++;
    if (line_number == 6) {
      // Save the position of the sixth line
      sixth_line_position = ftell(file) - strlen(line);
      break;
    }
  }

  if (line_number != 6) {
    fprintf(stderr, "Error: File does not have enough lines.\n");
    fclose(file);
    return -1;
  }

  // Move back to the start of the sixth line and overwrite it
  fseek(file, sixth_line_position, SEEK_SET);
  fprintf(file, "0 %d 0 0 # 8 ->latency optimal   9->rab mcpy   10->rab dt   11->rab single dt   12->rab segmented   13->rab contiguous\n", new_value);

  fclose(file);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <filename> <number between 0 and 16>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int new_value = atoi(argv[2]);
  const char *filename = argv[1];

  if (update_file(filename, new_value) != 0){
    return EXIT_FAILURE;
 }

  return EXIT_SUCCESS;
}
