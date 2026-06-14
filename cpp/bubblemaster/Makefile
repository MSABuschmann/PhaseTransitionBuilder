CXX      = CC
CXXFLAGS = -std=c++17 -O3 -fopenmp -ffast-math -Wall -Wextra -Wno-sign-compare

HDF5_INCLUDE = ${HDF5_DIR}/include
HDF5_LIB     = ${HDF5_DIR}/lib

INCLUDES = -I../../common -I$(HDF5_INCLUDE)
LIBS     = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 -lgsl -lgslcblas -lm -fopenmp

SRCDIR   = src
SRCS     = $(SRCDIR)/main.cpp \
           $(SRCDIR)/evolution.cpp \
           $(SRCDIR)/integrator.cpp \
           $(SRCDIR)/io.cpp
OBJS     = $(SRCS:.cpp=.o)
TARGET   = bubblemaster

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(SRCDIR)/main.cpp: $(SRCDIR)/evolution.h $(SRCDIR)/integrator.h $(SRCDIR)/io.h $(SRCDIR)/setup.h
$(SRCDIR)/evolution.cpp: $(SRCDIR)/evolution.h $(SRCDIR)/io.h
$(SRCDIR)/integrator.cpp: $(SRCDIR)/integrator.h $(SRCDIR)/integrand.h $(SRCDIR)/interpolator.h $(SRCDIR)/io.h
$(SRCDIR)/io.cpp: $(SRCDIR)/io.h

clean:
	rm -f $(OBJS) $(TARGET)
