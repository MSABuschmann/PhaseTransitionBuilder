SHELL = /bin/bash

BIN_DIR      = bin
BUBBLEMASTER = cpp/bubblemaster
SOLVER_1D    = cpp/solver_1d
WEIGHTS      = cpp/weights

.PHONY: all clean bubblemaster solver_1d weights

all: $(BIN_DIR) bubblemaster solver_1d weights
	@echo ""
	@echo "Build complete:"
	@echo "  $(BIN_DIR)/bubblemaster"
	@echo "  $(BIN_DIR)/solver_1d"
	@echo "  $(BIN_DIR)/weights"

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

bubblemaster: $(BIN_DIR)
	$(MAKE) -C $(BUBBLEMASTER)
	cp $(BUBBLEMASTER)/bubblemaster $(BIN_DIR)/bubblemaster

solver_1d: $(BIN_DIR)
	$(MAKE) -C $(SOLVER_1D)
	cp $(SOLVER_1D)/solver_1d $(BIN_DIR)/solver_1d

weights: $(BIN_DIR)
	$(MAKE) -C $(WEIGHTS)
	cp $(WEIGHTS)/weights $(BIN_DIR)/weights

clean:
	$(MAKE) -C $(BUBBLEMASTER) clean
	$(MAKE) -C $(SOLVER_1D) clean
	$(MAKE) -C $(WEIGHTS) clean
	rm -rf $(BIN_DIR)
