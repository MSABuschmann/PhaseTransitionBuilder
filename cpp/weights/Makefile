CXX      = g++
CXXFLAGS = -std=c++17 -O2 -fopenmp -Wall -Wextra
INCLUDES = -I../../common \
           -I/usr/include/hdf5/serial \
           -I/usr/local/include \
           $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")
LIBS     = -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
           -lhdf5_cpp -lhdf5 -lm -lgomp

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
