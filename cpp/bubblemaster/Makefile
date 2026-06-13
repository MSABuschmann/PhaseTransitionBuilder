CXX      = g++
CXXFLAGS = -std=c++17 -O2 -fopenmp -Wall -Wextra
INCLUDES = -I../../common -I/usr/include/hdf5/serial -I/usr/local/include
LIBS     = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5 -lgsl -lgslcblas -lm -lgomp

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
