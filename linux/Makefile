.PHONY: clean all

CC = gcc-5
CXX = g++-5
NVCC = nvcc
AR = ar
RANLIB = ranlib
XXD = xxd
CFLAGS = -Wall -O2
CXXFLAGS = -Wall -O2 -std=c++11
LDFLAGS = -Wall -O2 -std=c++11
NVCCFLAGS = -O2 -Wno-deprecated-gpu-targets -std=c++11
NVLDFLAGS = -O2 -Wno-deprecated-gpu-targets -std=c++11
#CFLAGS = -Wall -g
#CXXFLAGS = -Wall -g -std=c++11
#LDFLAGS = -Wall -g -std=c++11
#NVCCFLAGS = -g -Wno-deprecated-gpu-targets -std=c++11
#NVLDFLAGS = -g -Wno-deprecated-gpu-targets -std=c++11
NVLD = nvcc
LD = g++-5
INCDIRS = -I../enigma-cuda-lib
CUDAINCDIRS = -I/usr/local/cuda-8.0/include
CLPPINCDIRS = -I../clpp
OPENCLLIBDIR = -L/opt/amdgpu-pro/lib64
CLRXLIBDIR = /home/mat/docs/dev/usr/lib64
CLRXINCDIR = /home/mat/docs/dev/usr/include

LIB_SRCS = \
../enigma-cuda-lib/cuda_code.cu \
../enigma-cuda-lib/ini_file.cpp \
../enigma-cuda-lib/iterator.cpp \
../enigma-cuda-lib/key.cpp \
../enigma-cuda-lib/ngrams.cpp \
../enigma-cuda-lib/opencl_code.cpp \
../enigma-cuda-lib/plugboard.cpp \
../enigma-cuda-lib/runner.cpp \
../enigma-cuda-lib/segmenter.cpp \
../enigma-cuda-lib/settings.cpp \
../enigma-cuda-lib/test_helper.cu \
../enigma-cuda-lib/util.cpp \
../enigma-cuda-lib/wiring.cpp

PROG_SRCS = ../enigma-cuda/enigma-cuda.cpp

ALL_SRCS = $(LIB_SRCS) $(PROG_SRCS)

all: enigma-cuda enigma-opencl

libenigma-cuda.a: cuda_code.o ini_file.o iterator.o key.o ngrams.o plugboard.o \
		runner.o segmenter.o settings.o util.o wiring.o
	$(AR) cr $@ $^
	$(RANLIB) $@

libenigma-opencl.a: ini_file.o iterator.o key.o ngrams.o opencl_code.o plugboard.o \
		runner-ocl.o segmenter.o settings.o util.o wiring.o opencl_program_data.o \
		climb_clrx_data.o
	$(AR) cr $@ $^
	$(RANLIB) $@

enigma-cuda: enigma-cuda.o libenigma-cuda.a
	$(NVLD) $(NVLDFLAGS) -o $@ $^

enigma-opencl: enigma-cuda.o libenigma-opencl.a $(CLRXLIBDIR)/libCLRXAmdAsm.a \
		$(CLRXLIBDIR)/libCLRXAmdBin.a $(CLRXLIBDIR)/libCLRXUtils.a
	$(LD) $(LDFLAGS) -o $@ $^ $(OPENCLLIBDIR) -lOpenCL -ldl -pthread

%.o: ../enigma-cuda/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c -o $@ $<

%.o: ../enigma-cuda-lib/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c -o $@ $<

%-ocl.o: ../enigma-cuda-lib/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -DHAVE_OPENCL=1 -c -o $@ $<

%.o: ../enigma-cuda-lib/%.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAINCDIRS) $(INCDIRS) -c -o $@ $<

make.deps: $(ALL_SRCS)
	$(CXX) $(CXXFLAGS) $(INCDIRS) -I$(CLRXINCDIR) $(CUDAINCDIRS) $(CLPPINCDIRS) \
			-x c++ -M $(ALL_SRCS) \
	| sed -e "s/^\\(.*:.*\\)$$/\\1/" > make.deps
	$(CXX) $(CXXFLAGS) $(INCDIRS) $(CUDAINCDIRS) $(CLPPINCDIRS) -x c++ -M \
	../enigma-cuda-lib/runner.cpp \
	| sed -e "s/^.*:\\(.*\\)$$/runner-ocl.o: \\1/" >> make.deps

opencl_code.o: ../enigma-cuda-lib/opencl_code.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) $(CLPPINCDIRS) -I$(CLRXINCDIR) -c -o $@ $<

climb_clrx_data.c: ../enigma-cuda-lib/climb.clrx
	$(XXD) -i $< $@

opencl_program_data.c: ../enigma-cuda-lib/opencl_program.cl
	$(XXD) -i $< $@

opencl_program_data.o: opencl_program_data.c
	$(CC) $(CFLAGS) -c -o $@ $<

climb_clrx_data.o: climb_clrx_data.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f make.deps
	rm -f enigma-cuda enigma-opencl *.o
	rm -f libenigma-cuda.a libenigma-opencl.a
	rm -f opencl_program_data.c climb_clrx_data.c

include make.deps
