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

OBJS_GSL         = $(SRCS_GSL:.cpp=_gsl.o)
OBJS_FILON       = $(SRCS_FILON:.cpp=_filon.o)
OBJS_GPU         = $(SRCS_GPU:.cpp=_gpu.o) $(SRCDIR)/gpu_integrator_gpu.o
OBJS_GPU_PROFILE = $(SRCS_GPU:.cpp=_gpu_profile.o) $(SRCDIR)/gpu_integrator_gpu_profile.o

# Local macOS build (Homebrew clang + libomp + HDF5 + GSL)
CXX_LOCAL      = clang++
HDF5_LOCAL     = /opt/homebrew/opt/hdf5
GSL_LOCAL      = /opt/homebrew/opt/gsl
LIBOMP_LOCAL   = /opt/homebrew/opt/libomp
CXXFLAGS_LOCAL = -std=c++17 -O2 -Xpreprocessor -fopenmp -ffast-math -Wno-sign-compare \
                 -I$(LIBOMP_LOCAL)/include -I$(GSL_LOCAL)/include
LIBS_LOCAL     = -L$(HDF5_LOCAL)/lib -lhdf5_cpp -lhdf5 \
                 -L$(GSL_LOCAL)/lib -lgsl -lgslcblas \
                 -L$(LIBOMP_LOCAL)/lib -lomp -lm

.PHONY: all gsl filon gpu gpu_profile local clean

all: gsl filon gpu

gsl:         bubblemaster
filon:       bubblemaster_filon
gpu:         bubblemaster_gpu
gpu_profile: bubblemaster_gpu_profile

# --- GSL build ---
bubblemaster: $(OBJS_GSL)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS_GSL)

# --- Local macOS build: single-step compile+link, no separate .o files ---
.PHONY: local
local:
	$(CXX_LOCAL) $(CXXFLAGS_LOCAL) -I../../common -I$(HDF5_LOCAL)/include \
	    -o bubblemaster $(SRCS_GSL) $(LIBS_LOCAL)

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

# --- GPU profiling build ---
# Same as the gpu target, but with clock64()-based in-kernel timing
# instrumentation compiled in (GW_KERNEL_TIMING). Diagnostic only: works
# around DCGM blocking ncu/Nsight Compute on shared/exclusive nodes alike.
# The normal `gpu` target/binary is completely untouched by this flag.
bubblemaster_gpu_profile: $(OBJS_GPU_PROFILE)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS_GPU)

$(SRCDIR)/%_gpu_profile.o: $(SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DUSE_GPU -c -o $@ $<

$(SRCDIR)/gpu_integrator_gpu_profile.o: $(SRCDIR)/gpu_integrator.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DGW_KERNEL_TIMING -c -o $@ $<

clean:
	rm -f $(SRCDIR)/*_gsl.o $(SRCDIR)/*_filon.o $(SRCDIR)/*_gpu.o $(SRCDIR)/*_gpu_profile.o \
	      bubblemaster bubblemaster_filon bubblemaster_gpu bubblemaster_gpu_profile
