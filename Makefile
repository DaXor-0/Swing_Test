.PHONY: all clean libswing debug test ompi_rules

# Build all components
all: libswing debug test ompi_rules

# Build the libswing static library
libswing:
	@echo -e "\033[1;34m[BUILD] Compiling libswing static library...\033[0m"
	$(MAKE) -C libswing
	@echo ""

# Build the debug executable
debug: libswing
	@echo -e "\033[1;34m[BUILD] Compiling debug executable...\033[0m"
	$(MAKE) -C debug
	@echo ""

# Build the test executable
test: libswing
	@echo -e "\033[1;34m[BUILD] Compiling test executable...\033[0m"
	$(MAKE) -C test
	@echo ""

# Build the change_collective_rules program
ompi_rules:
	@echo -e "\033[1;34m[BUILD] Compiling ompi_rules executable...\033[0m"
	$(MAKE) -C ompi_rules
	@echo ""

# Clean all builds
clean:
	@echo -e "\033[1;31m[CLEAN] Cleaning all builds...\033[0m"
	$(MAKE) -C libswing clean
	$(MAKE) -C debug clean
	$(MAKE) -C test clean
	$(MAKE) -C ompi_rules clean
