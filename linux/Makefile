.PHONY: clean all

CC = gcc-5
CXX = g++-5
NVCC = nvcc
AR = ar
RANLIB = ranlib
CFLAGS = -Wall -O2
CXXFLAGS = -Wall -O2 -std=c++11
NVCCFLAGS = -O2 -Wno-deprecated-gpu-targets -std=c++11
#LDFLAGS = -O2 -Wno-deprecated-gpu-targets -std=c++11 --cudart shared
LDFLAGS = -O2 -Wno-deprecated-gpu-targets -std=c++11
LD = nvcc
INCDIRS = -I../enigma-cuda-lib -I/usr/local/cuda-8.0/include

LIB_SRCS = \
../enigma-cuda-lib/cuda_code.cu \
../enigma-cuda-lib/ini_file.cpp \
../enigma-cuda-lib/iterator.cpp \
../enigma-cuda-lib/key.cpp \
../enigma-cuda-lib/ngrams.cpp \
../enigma-cuda-lib/plugboard.cpp \
../enigma-cuda-lib/runner.cpp \
../enigma-cuda-lib/segmenter.cpp \
../enigma-cuda-lib/settings.cpp \
../enigma-cuda-lib/test_helper.cu \
../enigma-cuda-lib/util.cpp \
../enigma-cuda-lib/wiring.cpp

PROG_SRCS = ../enigma-cuda/enigma-cuda.cpp

ALL_SRCS = $(LIB_SRCS) $(PROG_SRCS)

all: enigma-cuda

libenigma-cuda.a: cuda_code.o ini_file.o iterator.o key.o ngrams.o plugboard.o \
		runner.o segmenter.o settings.o util.o wiring.o
	$(AR) cr $@ $^
	$(RANLIB) $@

enigma-cuda: enigma-cuda.o libenigma-cuda.a
	$(LD) $(LDFLAGS) -o $@ $^

%.o: ../enigma-cuda/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c -o $@ $<

%.o: ../enigma-cuda-lib/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c -o $@ $<

%.o: ../enigma-cuda-lib/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCDIRS) -c -o $@ $<

make.deps: $(ALL_SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCDIRS) -x c++ -M $(ALL_SRCS) | sed -e "s/^\\(.*:.*\\)$$/\\1/" > make.deps

enigma-cuda.o: ../enigma-cuda/enigma-cuda.cpp

cuda_code.o: ../enigma-cuda-lib/cuda_code.cu
ini_file.o: ../enigma-cuda-lib/ini_file.cpp
iterator.o: ../enigma-cuda-lib/iterator.cpp
key.o: ../enigma-cuda-lib/key.cpp
ngrams.o: ../enigma-cuda-lib/ngrams.cpp
plugboard.o: ../enigma-cuda-lib/plugboard.cpp
runner.o: ../enigma-cuda-lib/runner.cpp
segmenter.o: ../enigma-cuda-lib/segmenter.cpp
settings.o: ../enigma-cuda-lib/settings.cpp
#test_helper.o: ../enigma-cuda-lib/test_helper.cu
util.o: ../enigma-cuda-lib/util.cpp
wiring.o: ../enigma-cuda-lib/wiring.cpp

clean:
	rm -f make.deps
	rm -f enigma-cuda *.o
	rm -f libenigma-cuda.a

include make.deps
