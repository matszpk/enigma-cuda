WITH_CLRX = 1

CXX = cl
CLINCDIR = "c:\Program Files (x86)\AMD APP SDK\3.0\include"
CLLIBDIR = "c:\Program Files (x86)\AMD APP SDK\3.0\lib\x86_64"
!if $(WITH_CLRX)
CLRXINCDIR = /I"c:\matfiles\usr\include"
CLRXLIBDIR = "c:\matfiles\usr\lib"
CLRXDEFS = /DHAVE_CLRX=1
EXTRA_CLRXOBJS = climb_clrx_data.obj
!else
CLRXINCDIR = 
CLRXLIBDIR = 
CLRXDEFS =
EXTRA_CLRXOBJS =
!endif
XXD = "c:\Program Files (x86)\Vim\vim74\xxd"
CXXFLAGS =  /EHsc /W1 /O2 /I$(CLINCDIR) $(CLRXINCDIR) /I..\enigma-cuda-lib /I..\clpp
!if $(WITH_CLRX)
CLRXLIBS = $(CLRXLIBDIR)\CLRXAmdAsmStatic.lib $(CLRXLIBDIR)\CLRXAmdBinStatic.lib \
        $(CLRXLIBDIR)\CLRXUtilsStatic.lib
!else
CLRXLIBS =
!endif

all: enigma-opencl.exe

ENIGMA_OBJS = enigma-cuda.obj ini_file.obj iterator.obj key.obj ngrams.obj opencl_code.obj plugboard.obj \
		runner.obj segmenter.obj settings.obj util.obj wiring.obj opencl_program_data.obj $(EXTRA_CLRXOBJS)

enigma-opencl.exe: $(ENIGMA_OBJS)
        $(CXX) $(CXXFLAGS) /Feenigma-opencl.exe $** $(CLRXLIBS) $(CLLIBDIR)\OpenCL.lib shell32.lib user32.lib


enigma-cuda.obj: ..\enigma-cuda\enigma-cuda.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $** 

ini_file.obj: ..\enigma-cuda-lib\ini_file.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

iterator.obj: ..\enigma-cuda-lib\iterator.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

key.obj: ..\enigma-cuda-lib\key.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

opencl_code.obj: ..\enigma-cuda-lib\opencl_code.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CLRXDEFS) /DHAVE_OPENCL=1 /c /Fo$@ $**

ngrams.obj: ..\enigma-cuda-lib\ngrams.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

plugboard.obj: ..\enigma-cuda-lib\plugboard.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

runner.obj: ..\enigma-cuda-lib\runner.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /DHAVE_OPENCL=1 /c /Fo$@ $**

segmenter.obj: ..\enigma-cuda-lib\segmenter.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

settings.obj: ..\enigma-cuda-lib\settings.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

util.obj: ..\enigma-cuda-lib\util.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

wiring.obj: ..\enigma-cuda-lib\wiring.cpp
        $(CXX) $(CXXFLAGS) $(CPPFLAGS) /c /Fo$@ $**

opencl_program_data.c: ..\enigma-cuda-lib\opencl_program.cl
        $(XXD) -i $** $@

climb_clrx_data.c: ..\enigma-cuda-lib\climb.clrx
	$(XXD) -i $** $@

clean:
        del enigma-opencl.exe *.obj opencl_program_data.c climb_clrx_data.c