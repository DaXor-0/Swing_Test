# Compiler and flags
CFLAGS = ${CFLAGS_COMMON}

# Directories
OBJ_DIR = ../obj/lib
LIB_DIR = ../lib

# Sources and objects
LIB_SRC = $(wildcard *.c)
LIB_OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(LIB_SRC))
LIB_DEPS = $(patsubst %.c, $(OBJ_DIR)/%.d, $(LIB_SRC))
STATIC_LIB = $(LIB_DIR)/libswing.a

# Default target
all: $(STATIC_LIB)

# Build static library
$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(STATIC_LIB)$(NC)"
	ar rcs $@ $^

# Compile object files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -c $< -o $@

-include $(LIB_DEPS)

# Clean target
clean:
	@echo -e "[CLEAN] Removing object files and library..."
	@rm -rf $(OBJ_DIR) $(STATIC_LIB)
