UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  BREW         := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
  CXX          = $(BREW)/opt/llvm/bin/clang++
  HDF5_INCLUDE = $(BREW)/opt/hdf5/include
  HDF5_LIB     = $(BREW)/opt/hdf5/lib
  EIGEN_INC    = $(shell find $(BREW)/Cellar/eigen -name "eigen3" -type d 2>/dev/null | head -1)
  GSL_INCLUDE  = $(BREW)/opt/gsl/include
  GSL_LIB      = $(BREW)/opt/gsl/lib
  OMP_INC      = $(BREW)/opt/libomp/include
  OMP_LIB      = $(BREW)/opt/libomp/lib
  OMP_FLAGS    = -I$(OMP_INC) -L$(OMP_LIB)
else
  CXX          = CC
  HDF5_INCLUDE = $(HDF5_DIR)/include
  HDF5_LIB     = $(HDF5_DIR)/lib
  EIGEN_INC    = $(or $(shell pkg-config --variable=includedir eigen3 2>/dev/null),/usr/include)/eigen3
  GSL_INCLUDE  =
  GSL_LIB      =
  OMP_FLAGS    =
  # NERSC: `module load cray-gsl` before building -- the CC wrapper picks up
  # its include/lib paths automatically (same assumption cpp/bubblemaster's
  # GSL variant already makes).
endif

# No -ffast-math here (unlike the other cpp/ binaries): the phase
# computation does cos(w*proj)/sin(w*proj) with arguments that can reach
# into the thousands (w up to ~14, |proj| up to a few hundred), and
# -ffast-math can substitute a faster/lower-accuracy vectorized trig
# implementation there -- compiler/libm/hardware dependent, so something
# that's harmless on one toolchain can be a real several-percent error on
# another. This binary's hot loop is otherwise just interpolation
# arithmetic, which doesn't need -ffast-math's other optimizations.
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra -Wno-sign-compare $(OMP_FLAGS)

INCLUDES = -I../../common \
           -I$(HDF5_INCLUDE) \
           -I$(EIGEN_INC) \
           $(if $(GSL_INCLUDE),-I$(GSL_INCLUDE))
LIBS     = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 \
           $(if $(GSL_LIB),-L$(GSL_LIB)) -lgsl -lgslcblas \
           -lm -fopenmp $(OMP_FLAGS)

SRCDIR   = src
SRCS     = $(SRCDIR)/main.cpp \
           $(SRCDIR)/io.cpp \
           $(SRCDIR)/geometry.cpp \
           $(SRCDIR)/quadrature.cpp \
           $(SRCDIR)/amplitude_table.cpp \
           $(SRCDIR)/coherent_sum.cpp
OBJS     = $(SRCS:.cpp=.o)
TARGET   = coherent

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
