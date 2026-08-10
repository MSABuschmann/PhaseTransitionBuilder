SHELL = /bin/bash

BIN_DIR      = bin
BUBBLEMASTER = cpp/bubblemaster
SOLVER_1D    = cpp/solver_1d
WEIGHTS      = cpp/weights
COHERENT     = cpp/coherent

PYTHON   := python3
PY_EXT   := $(shell $(PYTHON)-config --extension-suffix)
PY_INC   := $(shell $(PYTHON) -m pybind11 --includes)

.PHONY: all clean bubblemaster bubblemaster_filon bubblemaster_gpu solver_1d weights coherent _potential

all: $(BIN_DIR) _potential bubblemaster bubblemaster_filon solver_1d weights coherent
	@echo ""
	@echo "Build complete (CPU targets):"
	@echo "  ptbuilder/_potential$(PY_EXT)"
	@echo "  $(BIN_DIR)/bubblemaster"
	@echo "  $(BIN_DIR)/bubblemaster_filon"
	@echo "  $(BIN_DIR)/solver_1d"
	@echo "  $(BIN_DIR)/weights"
	@echo "  $(BIN_DIR)/coherent"
	@echo "Run 'make bubblemaster_gpu' separately (requires cudatoolkit module)"

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

bubblemaster: $(BIN_DIR)
	$(MAKE) -C $(BUBBLEMASTER) gsl
	cp $(BUBBLEMASTER)/bubblemaster $(BIN_DIR)/bubblemaster

bubblemaster_filon: $(BIN_DIR)
	$(MAKE) -C $(BUBBLEMASTER) filon
	cp $(BUBBLEMASTER)/bubblemaster_filon $(BIN_DIR)/bubblemaster_filon

bubblemaster_gpu: $(BIN_DIR)
	$(MAKE) -C $(BUBBLEMASTER) gpu
	cp $(BUBBLEMASTER)/bubblemaster_gpu $(BIN_DIR)/bubblemaster_gpu


solver_1d: $(BIN_DIR)
	$(MAKE) -C $(SOLVER_1D)
	cp $(SOLVER_1D)/solver_1d $(BIN_DIR)/solver_1d

weights: $(BIN_DIR)
	$(MAKE) -C $(WEIGHTS)
	cp $(WEIGHTS)/weights $(BIN_DIR)/weights

coherent: $(BIN_DIR)
	$(MAKE) -C $(COHERENT)
	cp $(COHERENT)/coherent $(BIN_DIR)/coherent

_potential: ptbuilder/_potential$(PY_EXT)

ptbuilder/_potential$(PY_EXT): cpp/ptbuilder/_potential.cpp cpp/common/potential.h
	$(CXX) -O2 -shared -fPIC -std=c++17 $(PY_INC) -o $@ $<

clean:
	$(MAKE) -C $(BUBBLEMASTER) clean
	$(MAKE) -C $(SOLVER_1D) clean
	$(MAKE) -C $(WEIGHTS) clean
	$(MAKE) -C $(COHERENT) clean
	rm -rf $(BIN_DIR)
	rm -f ptbuilder/_potential*.so
