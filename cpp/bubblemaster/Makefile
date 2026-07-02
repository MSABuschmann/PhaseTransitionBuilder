CXX      = CC
CXXFLAGS = -std=c++17 -O3 -fopenmp -ffast-math -Wall -Wextra -Wno-sign-compare

HDF5_INCLUDE = ${HDF5_DIR}/include
HDF5_LIB     = ${HDF5_DIR}/lib

INCLUDES = -I../../common -I$(HDF5_INCLUDE)

# GSL version
LIBS_GSL  = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 -lgsl -lgslcblas -lm -fopenmp
# Filon version — no GSL dependency
LIBS_FILON = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 -lm -fopenmp

SRCDIR = src

SRCS_COMMON = $(SRCDIR)/main.cpp \
              $(SRCDIR)/evolution.cpp \
              $(SRCDIR)/io.cpp

SRCS_GSL   = $(SRCS_COMMON) $(SRCDIR)/integrator.cpp
SRCS_FILON = $(SRCS_COMMON) $(SRCDIR)/filon_integrator.cpp

OBJS_GSL   = $(SRCS_GSL:.cpp=_gsl.o)
OBJS_FILON = $(SRCS_FILON:.cpp=_filon.o)

.PHONY: all gsl filon clean

all: gsl filon

gsl: bubblemaster
filon: bubblemaster_filon

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

clean:
	rm -f $(SRCDIR)/*_gsl.o $(SRCDIR)/*_filon.o bubblemaster bubblemaster_filon
