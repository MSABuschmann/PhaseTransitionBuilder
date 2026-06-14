CXX      = CC
CXXFLAGS = -std=c++17 -O3 -fopenmp -ffast-math -Wall -Wextra -Wno-sign-compare

HDF5_INCLUDE = ${HDF5_DIR}/include
HDF5_LIB     = ${HDF5_DIR}/lib

INCLUDES = -I../../common \
           -I$(HDF5_INCLUDE) \
           $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/local/include/eigen3")
LIBS     = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 -lm -fopenmp

SRCDIR   = src
SRCS     = $(SRCDIR)/main.cpp \
           $(SRCDIR)/interaction.cpp
OBJS     = $(SRCS:.cpp=.o)
TARGET   = weights

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
