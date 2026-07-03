CXX      = CC
CXXFLAGS = -std=c++17 -O3 -fopenmp -ffast-math -Wall -Wextra -Wno-sign-compare

# CUDA compiler — requires cudatoolkit module on Perlmutter
NVCC       = nvcc
# sm_80 = A100 (Perlmutter GPU nodes); adjust for other GPUs
NVCCFLAGS  = -std=c++17 -O3 -arch=sm_80 --expt-relaxed-constexpr \
             -Xcompiler -fopenmp -Xcompiler -ffast-math

HDF5_INCLUDE = ${HDF5_DIR}/include
HDF5_LIB     = ${HDF5_DIR}/lib

INCLUDES = -I../../common -I$(HDF5_INCLUDE)

# GSL version
LIBS_GSL   = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 -lgsl -lgslcblas -lm -fopenmp
# Filon version — no GSL dependency
LIBS_FILON = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 -lm -fopenmp
# GPU version — nvcc links cudart automatically
LIBS_GPU   = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 -lm -Xcompiler -fopenmp

SRCDIR = src

SRCS_COMMON = $(SRCDIR)/main.cpp \
              $(SRCDIR)/evolution.cpp \
              $(SRCDIR)/io.cpp

SRCS_GSL   = $(SRCS_COMMON) $(SRCDIR)/integrator.cpp
SRCS_FILON = $(SRCS_COMMON) $(SRCDIR)/filon_integrator.cpp
SRCS_GPU   = $(SRCS_COMMON)   # .cu compiled separately below

OBJS_GSL   = $(SRCS_GSL:.cpp=_gsl.o)
OBJS_FILON = $(SRCS_FILON:.cpp=_filon.o)
OBJS_GPU   = $(SRCS_GPU:.cpp=_gpu.o) $(SRCDIR)/gpu_integrator_gpu.o

.PHONY: all gsl filon gpu clean

all: gsl filon gpu

gsl:   bubblemaster
filon: bubblemaster_filon
gpu:   bubblemaster_gpu

# --- GSL build ---
bubblemaster: $(OBJS_GSL)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS_GSL)

$(SRCDIR)/%_gsl.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# --- Filon build ---
bubblemaster_filon: $(OBJS_FILON)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS_FILON)

$(SRCDIR)/%_filon.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DUSE_FILON -c -o $@ $<

# --- GPU build ---
bubblemaster_gpu: $(OBJS_GPU)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS_GPU)

$(SRCDIR)/%_gpu.o: $(SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DUSE_GPU -c -o $@ $<

$(SRCDIR)/gpu_integrator_gpu.o: $(SRCDIR)/gpu_integrator.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(SRCDIR)/*_gsl.o $(SRCDIR)/*_filon.o $(SRCDIR)/*_gpu.o \
	      bubblemaster bubblemaster_filon bubblemaster_gpu
