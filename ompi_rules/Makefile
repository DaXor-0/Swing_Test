# Compiler and flags
CC = gcc
CFLAGS = -g -Wall

# Directories
SRC_FILE = change_collective_rules.c
BIN_DIR = ../bin

# Output
OUTPUT = $(BIN_DIR)/change_collective_rules

# Default target
all: $(OUTPUT)

# Build the executable
$(OUTPUT): $(SRC_FILE)
	@mkdir -p $(BIN_DIR)
	@echo -e "\033[1;33m[CC] Compiling $< -> $@\033[0m"
	$(CC) $(CFLAGS) $< -o $@

# Clean target
clean:
	@echo -e "\033[1;31m[CLEAN] Removing ompi_rules binary...\033[0m"
	rm -rf $(OUTPUT)
