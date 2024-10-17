MPICC = mpicc
CFLAGS_MPI = -g -O3 -Wall -pedantic -lm

GCC = gcc

SRC_DIR = src
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# Executable names
MAIN_EXEC = out
RULES_EXEC = update_collective_rules
DEBUG_EXEC = debug

# Main target that builds both executables
all: $(MAIN_EXEC) $(RULES_EXEC) $(DEBUG_EXEC)

# Build the main test executable with mpicc
$(MAIN_EXEC): $(OBJS)
	$(MPICC) $(CFLAGS_MPI) $(OBJS) -o $(MAIN_EXEC)

# Build object files for the source files in the src directory with mpicc
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(MPICC) $(CFLAGS_MPI) -c $< -o $@

$(DEBUG_EXEC): debug.c
	$(MPICC) $(CFLAGS_MPI) debug.c -o $(DEBUG_EXEC)

# Compile update_collective_rules.c into its own executable with gcc and different flags
$(RULES_EXEC): update_collective_rules.c
	$(GCC) update_collective_rules.c -o $(RULES_EXEC)


# Clean command that removes object files and both executables
clean:
	rm -rf $(OBJ_DIR) $(MAIN_EXEC) $(RULES_EXEC) $(DEBUG_EXEC)
