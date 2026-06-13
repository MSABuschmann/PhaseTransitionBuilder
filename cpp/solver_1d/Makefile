CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I../../common -I/usr/include/hdf5/serial -I/usr/local/include
LIBS     = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5 -lm

SRCDIR   = src
SRCS     = $(SRCDIR)/main.cpp \
           $(SRCDIR)/solver.cpp \
           $(SRCDIR)/io.cpp
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
