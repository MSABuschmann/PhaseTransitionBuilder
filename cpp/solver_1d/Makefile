CXX      = CC
CXXFLAGS = -std=c++17 -O3 -ffast-math -Wall -Wextra -Wno-sign-compare

HDF5_INCLUDE = ${HDF5_DIR}/include
HDF5_LIB     = ${HDF5_DIR}/lib

INCLUDES = -I../../common -I$(HDF5_INCLUDE)
LIBS     = -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 -lm

SRCDIR   = src
SRCS     = $(SRCDIR)/main.cpp \
           $(SRCDIR)/solver.cpp
OBJS     = $(SRCS:.cpp=.o)
TARGET   = solver_1d

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
