#include <stdio.h>
#include <stdlib.h>

#define FILENAME "collective_rules.txt"
#define LINE_NUMBER 6

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <new_number>\n", argv[0]);
        return 1;
    }

    int new_number = atoi(argv[1]);
    if (new_number < 1 || new_number > 12) {
        fprintf(stderr, "Error: Number must be between 1 and 12.\n");
        return 1;
    }

    FILE *file = fopen(FILENAME, "r+");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    char buffer[256];
    int line_counter = 0;
    long line_pos = 0;

    // Find the position of the 6th line in the file
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        line_counter++;
        if (line_counter == LINE_NUMBER) {
            line_pos = ftell(file);
            break;
        }
    }

    if (line_counter != LINE_NUMBER) {
        fprintf(stderr, "Error: File does not have enough lines.\n");
        fclose(file);
        return 1;
    }

    // Move the file pointer back to the beginning of the 6th line
    fseek(file, line_pos - sizeof(buffer), SEEK_SET);
    
    // Replace the 6th line with the new number
    fprintf(file, "0 %d 0 0  # 8 ->latency optimal   9->rab mcpy   10->rab dt   11->rab single dt   12->rab segmented\n", new_number);

    fclose(file);
    return 0;
}
