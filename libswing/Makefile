# Compiler and flags
MPICC = mpicc
CFLAGS = -O3 -Wall -I../include

# Directories
OBJ_DIR = ../obj/lib
LIB_DIR = ../lib

# Sources and objects
LIB_SRC = $(wildcard *.c)
LIB_OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(LIB_SRC))
STATIC_LIB = $(LIB_DIR)/libswing.a

# Default target
all: $(STATIC_LIB)

# Build static library
$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo -e "\033[1;32m[AR] Creating static library: $(STATIC_LIB)\033[0m"
	ar rcs $@ $^

# Compile object files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "\033[1;33m[MPICC] Compiling $< -> $@\033[0m"
	$(MPICC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	@echo -e "\033[1;31m[CLEAN] Removing object files and library...\033[0m"
	rm -rf $(OBJ_DIR) $(STATIC_LIB)
