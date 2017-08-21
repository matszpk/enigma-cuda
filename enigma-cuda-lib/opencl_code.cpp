/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */
/* Copyright (c) 2017 Mateusz Szpakowski                               */

#define __CLPP_CL_ABI_VERSION 101
#define __CLPP_CL_VERSION 101
#define __CLPP_CL_EXT 1

#include <stdexcept>
#include <iostream>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <memory>
#include <clpp.h>
#ifdef HAVE_CLRX
#include <CLRX/utils/GPUId.h>
#include <CLRX/amdasm/Assembler.h>
#endif
#include "opencl_code.h"
#include "plugboard.h"
#include "ngrams.h"
#include "iterator.h"

#ifdef _WIN32
#define snprintf _snprintf
#endif

#define REDUCE_MAX_THREADS 256
#define SCRAMBLER_STRIDE 8

#ifndef CL_DEVICE_BOARD_NAME_AMD
#define CL_DEVICE_BOARD_NAME_AMD                    0x4038
#endif

//#define DEBUG_CLIMB 1
#define DEBUG_DUMP 0
//#define DEBUG_RESULTS 1

// comment when application must accept any OpenCL platform
#define ACCEPT_ONLY_PREFERRED_PLATFORM 1
// for AMD
#define PLATFORM_VENDOR "Advanced Micro Devices, Inc."
// for NVIDIA
//#define PLATFORM_VENDOR "NVIDIA Corporation"

static int CLRX_GroupSize = 0;

static clpp::Device oclDevice;
static clpp::Context oclContext;
static clpp::Program oclProgram;
static clpp::Program oclClrxProgram;
static clpp::CommandQueue oclCmdQueue;
static clpp::Kernel GenerateScramblerKernel;
static size_t GenerateScramblerKernelWGSize;
static clpp::Kernel ClimbKernel;
static size_t ClimbKernelWGSize;
static clpp::Kernel FindBestResultKernel;
static clpp::Kernel FindBestResultKernel2;
static size_t FindBestResultKernelWGSize;
static cl_uint thBlockShift = 0;
#ifdef HAVE_CLRX
static bool useClrxAssembly = false;
#else
static const bool useClrxAssembly = false;
#endif

// buffers
static clpp::Buffer d_ciphertextBuffer;
static clpp::Buffer d_wiringBuffer;
static clpp::Buffer d_keyBuffer;
static size_t scramblerDataPitch;
static clpp::Buffer scramblerDataBuffer;
static clpp::Buffer d_orderBuffer;
static clpp::Buffer d_plugsBuffer;
static clpp::Buffer d_fixedBuffer;
#ifdef HAVE_CLRX
static cl_int d_fixedValue = 0;
#endif
static clpp::Buffer d_tempBuffer;
static clpp::Buffer d_unigramsBuffer;
static clpp::Buffer d_bigramsBuffer;
static size_t d_tempSize = 0;
static clpp::Buffer resultsBuffer;
static size_t trigramsBufferPitch;
static clpp::Buffer trigramsBuffer;

static bool shortBigrams = false;
static bool shortTrigrams = false;

#ifdef HAVE_CLRX
using namespace CLRX;
#endif

extern "C"
{

int8_t mod26(const int16_t x)
{
  return (ALPSIZE * 2 + x) % ALPSIZE;
}

};

#ifdef DEBUG_CLIMB
static int debugPart = -1; // full test
  #if DEBUG_DUMP&1
static clpp::Buffer sregDumpBuffer;
  #endif
  #if DEBUG_DUMP&2
static clpp::Buffer vregDumpBuffer;
  #endif

struct dim3
{
    size_t x, y, z;
};

static int8_t* scramblerData;
static int8_t d_ciphertext[MAX_MESSAGE_LENGTH];
static Wiring d_wiring;
static Key d_key;
static NGRAM_DATA_TYPE d_unigrams[ALPSIZE];
static NGRAM_DATA_TYPE d_bigrams[ALPSIZE][ALPSIZE];
static NGRAM_DATA_TYPE* trigramsData = NULL;
static int8_t d_order[ALPSIZE];
static int8_t d_plugs[ALPSIZE];
static bool d_fixed[ALPSIZE];

  #if DEBUG_DUMP&1
static cl_uint* expectedSRegs = NULL;
  #endif
  #if DEBUG_DUMP&2
static cl_uint* expectedVRegs = NULL;
  #endif

Task task;
#endif

void SetUpScramblerMemory()
{
  oclCmdQueue.writeBuffer(d_wiringBuffer, 0, sizeof(Wiring), &wiring);
  scramblerDataPitch = (28 + 15) & ~size_t(15);
  scramblerDataBuffer = clpp::Buffer(oclContext, CL_MEM_READ_WRITE,
                  scramblerDataPitch*ALPSIZE_TO3);
#ifdef DEBUG_CLIMB
  d_wiring = wiring;
  scramblerData = (int8_t*)::malloc(scramblerDataPitch*ALPSIZE_TO3);
  task.scrambler.pitch = scramblerDataPitch;
  task.scrambler.data = scramblerData;
#endif
  GenerateScramblerKernel.setArg(4, cl_uint(scramblerDataPitch));
  GenerateScramblerKernel.setArg(5, scramblerDataBuffer);
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
    ClimbKernel.setArg(3, cl_uint(scramblerDataPitch));
  ClimbKernel.setArg(4 - int(useClrxAssembly)*2, scramblerDataBuffer);
}

#ifdef DEBUG_CLIMB
void GenerateScramblerKernelHost(dim3 blockIdx, dim3 threadIdx)
{
  const int8_t * reflector;

  const int8_t * g_rotor;
  const int8_t * l_rotor;
  const int8_t * m_rotor;
  const int8_t * r_rotor;

  const int8_t * g_rev_rotor;
  const int8_t * l_rev_rotor;
  const int8_t * m_rev_rotor;
  const int8_t * r_rev_rotor;

  int8_t r_core_position;
  int8_t m_core_position;
  int8_t l_core_position;
  int8_t g_core_position;

  int8_t * entry;

  //if (threadIdx.x == 0)
  {
    //wirings
    reflector = d_wiring.reflectors[d_key.stru.ukwnum];

    g_rotor = d_wiring.rotors[d_key.stru.g_slot];
    l_rotor = d_wiring.rotors[d_key.stru.l_slot];
    m_rotor = d_wiring.rotors[d_key.stru.m_slot];
    r_rotor = d_wiring.rotors[d_key.stru.r_slot];

    g_rev_rotor = d_wiring.reverse_rotors[d_key.stru.g_slot];
    l_rev_rotor = d_wiring.reverse_rotors[d_key.stru.l_slot];
    m_rev_rotor = d_wiring.reverse_rotors[d_key.stru.m_slot];
    r_rev_rotor = d_wiring.reverse_rotors[d_key.stru.r_slot];

    //core positions
    r_core_position = blockIdx.x;
    m_core_position = blockIdx.y;
    l_core_position = blockIdx.z;
    g_core_position = mod26(d_key.sett.g_mesg - d_key.sett.g_ring);

    //address of scrambler entry
    entry = task.scrambler.data + task.scrambler.pitch * (
      l_core_position *  ALPSIZE * ALPSIZE +
      m_core_position *  ALPSIZE +
      r_core_position);
  }
  //__syncthreads();

  //scramble one char
  int8_t ch_in = threadIdx.x;
  int8_t ch_out = ch_in;

  ch_out = r_rotor[mod26(ch_out + r_core_position)] - r_core_position;
  ch_out = m_rotor[mod26(ch_out + m_core_position)] - m_core_position;
  ch_out = l_rotor[mod26(ch_out + l_core_position)] - l_core_position;

  if (d_key.stru.model == enigmaM4)
  {
    ch_out = g_rotor[mod26(ch_out + g_core_position)] - g_core_position;
    ch_out = reflector[mod26(ch_out)];
    ch_out = g_rev_rotor[mod26(ch_out + g_core_position)] - g_core_position;
  }
  else
  {
    ch_out = reflector[mod26(ch_out)];
  }

  ch_out = l_rev_rotor[mod26(ch_out + l_core_position)] - l_core_position;
  ch_out = m_rev_rotor[mod26(ch_out + m_core_position)] - m_core_position;
  ch_out = r_rev_rotor[mod26(ch_out + r_core_position)] - r_core_position;

  //char to scrambler
  entry[ch_in] = mod26(ch_out);
}
#endif

void GenerateScrambler(const Key & key)
{
#ifdef DEBUG_CLIMB
  std::cout << "\nGenerateScrambler call\n" << std::endl;
  { // clear buffer before generating
    clpp::BufferMapping mapping(oclCmdQueue, scramblerDataBuffer, true, CL_MAP_WRITE,
                        0, scramblerDataPitch*ALPSIZE_TO3);
    ::memset(mapping.get(), 0, scramblerDataPitch*ALPSIZE_TO3);
  }
  ::memset(task.scrambler.data, 0, scramblerDataPitch*ALPSIZE_TO3);
#endif
  oclCmdQueue.enqueueWriteBuffer(d_keyBuffer, 0, sizeof(Key), &key);
  clpp::Size3  dimGrid(ALPSIZE>>thBlockShift, ALPSIZE, ALPSIZE);
  clpp::Size3 dimBlock(GenerateScramblerKernelWGSize, 1, 1);
  dimGrid[0] *= dimBlock[0];
  oclCmdQueue.enqueueNDRangeKernel(GenerateScramblerKernel, dimGrid, dimBlock).wait();
#ifdef DEBUG_CLIMB
  d_key = key;
  // for comparison
  dim3 threadIdx = { 0, 0 ,0 };
  dim3 blockIdx = { 0, 0, 0 };
  for (blockIdx.z = 0; blockIdx.z < ALPSIZE; blockIdx.z++)
      for (blockIdx.y = 0; blockIdx.y < ALPSIZE; blockIdx.y++)
          for (blockIdx.x = 0; blockIdx.x < ALPSIZE; blockIdx.x++)
              for (threadIdx.x = 0; threadIdx.x < ALPSIZE; threadIdx.x++)
                  GenerateScramblerKernelHost(blockIdx, threadIdx);
#endif
}

std::string trimSpaces(const std::string& s)
{
    std::string::size_type pos = s.find_first_not_of(" \n\t\r\v\f");
    if (pos == std::string::npos)
        return "";
    std::string::size_type endPos = s.find_last_not_of(" \n\t\r\v\f");
    return s.substr(pos, endPos+1-pos);
}

static int OpenCL_turnover_modes = 0;
static int OpenCL_score_kinds = 0;
static int OpenCL_cipher_length = 0;

void setUpConfig(int turnover_modes, int score_kinds, int cipher_length)
{
#ifdef DEBUG_CLIMB
  std::cout << "turnover_modes: " << turnover_modes << ", score_kinds: " <<
          score_kinds << std::endl;
#endif
  OpenCL_turnover_modes = turnover_modes;
  OpenCL_score_kinds = score_kinds;
  OpenCL_cipher_length = cipher_length;
  CLRX_GroupSize = (cipher_length + 63) & ~63;
}

#ifdef HAVE_CLRX
static bool prepareAssemblyOfClimbKernel()
{
  GPUDeviceType devType = GPUDeviceType::CAPE_VERDE;
  try
  {
    std::string deviceName = oclDevice.getName();
    deviceName = trimSpaces(deviceName);
    devType = getGPUDeviceTypeFromName(deviceName.c_str());
  }
  catch(const CLRX::Exception& ex)
  { return false; }
  GPUArchitecture arch = getGPUArchitectureFromDeviceType(devType);
  
  bool useCL1 = false;
  {
    const char* clrxClimbLegacy = ::getenv("ECLRXCLIMB_LEGACY");
    useCL1 = clrxClimbLegacy!=NULL && ::strcmp(clrxClimbLegacy, "1")==0;
  }
  
  cxuint amdappVersion = 0;
  {
    const std::string driverVersion = oclDevice.getDriverVersion();
    cxuint major, minor;
    sscanf(driverVersion.c_str(), "%u.%u", &major, &minor);
    amdappVersion = major*100+minor;
  }
  if (amdappVersion < 138400)
    return false; // old driver not supported
  BinaryFormat binaryFormat = BinaryFormat::AMD;
  bool defaultCL2ForDriver = false;
  if (amdappVersion >= 200406 && !useCL1)
    defaultCL2ForDriver = true;
  
  if (defaultCL2ForDriver && arch >= GPUArchitecture::GCN1_1)
    binaryFormat = BinaryFormat::AMDCL2;
  
  const cl_uint bits = oclDevice.getAddressBits();
  // try to compile code
  Array<cxbyte> binary;
  const char* asmSource = (const char*)___enigma_cuda_lib_climb_clrx;
  size_t asmSourceSize = ___enigma_cuda_lib_climb_clrx_len;
  {
    ArrayIStream astream(asmSourceSize, asmSource);
    Assembler assembler("", astream, 0, binaryFormat, devType, std::cerr, std::cerr);
    assembler.set64Bit(bits==64);
    assembler.setDriverVersion(amdappVersion);
    assembler.addInitialDefSym("SCRAMBLER_PITCH", (28 + 15) & ~size_t(15));
    assembler.addInitialDefSym("SCRAMBLER_STRIDE", SCRAMBLER_STRIDE);
    assembler.addInitialDefSym("TRIGRAMS_PITCH",
              (shortTrigrams ? sizeof(cl_ushort) : sizeof(NGRAM_DATA_TYPE)) * ALPSIZE);
    assembler.addInitialDefSym("SCORE_KINDS", OpenCL_score_kinds);
    assembler.addInitialDefSym("TURNOVER_MODES", OpenCL_turnover_modes);
    assembler.addInitialDefSym("TASK_SIZE", OpenCL_cipher_length);
    assembler.addInitialDefSym("SHORT_BIGRAMS", shortBigrams);
    assembler.addInitialDefSym("SHORT_TRIGRAMS", shortTrigrams);
#ifdef DEBUG_CLIMB
    if (debugPart!=-1)
      assembler.addInitialDefSym("DEBUG", DEBUG_DUMP);
    else
      assembler.addInitialDefSym("DEBUG", 0);
    assembler.addInitialDefSym("DEBUG_PART", debugPart);
#else
    assembler.addInitialDefSym("DEBUG", 0);
#endif
    assembler.assemble();
    assembler.writeBinary(binary);
  }
  
  // create program and build
  oclClrxProgram = clpp::Program(oclContext, oclDevice, binary.size(), binary.data());
  if (useCL1 && amdappVersion >= 200406)
    oclClrxProgram.build("-legacy");
  else if (binaryFormat == BinaryFormat::AMDCL2 && amdappVersion < 200406)
    oclClrxProgram.build("-cl-std=CL2.0");
  else // default
    oclClrxProgram.build();
  ClimbKernel = clpp::Kernel(oclClrxProgram, "ClimbKernel");
  return true;
}
#endif

static const char* tempProgramSource =
R"ffSaCD(kernel void vectorAdd(uint n, const global float* a, const global float* b,
            global float* c)
{
    uint i = get_global_id(0);
    if (i >= n) return;
    c[i] = a[i] + b[i];
})ffSaCD";

static size_t getWorkGroupSizeMultiple()
{
  const clpp::Program tmpProg(oclContext, tempProgramSource);
  tmpProg.build();
  const clpp::Kernel tmpKernel(tmpProg, "vectorAdd");
  return tmpKernel.getPreferredWorkGroupSizeMultiple(oclDevice);
}

/*
 * OpenCL init
 */
bool SelectGpuDevice(int req_major, int req_minor, int settings_device, bool silent)
{
  std::vector<clpp::Platform> platforms = clpp::Platform::get();
  if (platforms.empty())
  {
    std::cerr << "OpenCL platform not found. Terminating..." << std::endl;
    return false;
  }
  int best_device = 0;
  int platformIndex = -1;
  for (size_t i = 0;  i < platforms.size(); i++)
  {
    std::string vendor = platforms[i].getVendor();
    vendor = trimSpaces(vendor);
    if (vendor==PLATFORM_VENDOR)
    {
      platformIndex = i;
      break;
    }
  }
  if (platformIndex==-1)
  {
#ifdef ACCEPT_ONLY_PREFERRED_PLATFORM
    std::cerr << "Preferred OpenCL platform vendor (" << PLATFORM_VENDOR <<
          ") not found." << std::endl;
    return false; // not found
#else
    std::cerr << "Preferred OpenCL platform vendor (" << PLATFORM_VENDOR << ") not found.\r\n"
        "Use first OpenCL platform\r\n";
    platformIndex = 0;
#endif
  }
  const clpp::Platform& platform = platforms[platformIndex];
  const std::vector<clpp::Device> devices = platform.getGPUDevices();

  if (devices.empty())
  {
    std::cerr << "OpenCL device not found. Terminating..." << std::endl;
    return false;
  }
  
  const char* cldevStr = ::getenv("CLDEV");
  if (settings_device != -1)
    best_device = settings_device;
  else if (cldevStr != nullptr)
    best_device = atoi(cldevStr);
  else if (devices.size() > 1)
  { // just select
    int max_cu = 0;
    for (size_t i = 0; i < devices.size(); i++)
    {
      const clpp::Device& device = devices[i];
      int cusNum = device.getMaxComputeUnits();
      if (cusNum >= max_cu)
      {
        best_device = i;
        max_cu = cusNum;
      }
    }
  }
  if (best_device < 0 || size_t(best_device) >= devices.size())
  {
    std::cerr << "Choosen device out of range" << std::endl;
    return false;
  }
  const clpp::Device& device = devices[best_device];
  if (!silent)
  {
    std::string boardName;
    try
    {
      std::string vendor = platform.getVendor();
      vendor = trimSpaces(vendor);
      if (vendor == "Advanced Micro Devices, Inc.")
      {
        boardName = device.getInfoString(CL_DEVICE_BOARD_NAME_AMD);
        boardName = trimSpaces(boardName);
      }
    }
    catch(...)
    { }
    
    std::string deviceName = device.getName();
    deviceName = trimSpaces(deviceName);
    std::cerr << "Found GPU " << best_device << ": '" << deviceName;
    if (!boardName.empty())
      std::cerr << "', Board name: '" << boardName;
    std::cerr << "', Compute units: " << device.getMaxComputeUnits() << std::endl;
  }
  oclDevice = device;
  
  
#ifdef DEBUG_CLIMB
  std::cout << "CipherLength: " << OpenCL_cipher_length << std::endl;
  {
    const char* debugPartStr = ::getenv("ECLIMB_DEBUG_PART");
    if (debugPartStr!=NULL && *debugPartStr!=0)
      debugPart = atoi(debugPartStr);
  }
  std::cout << "DebugPartClimb: " << debugPart << std::endl;
#endif
  
  /*
   * creating opencl stuff
   * building program and creating kernels
   */
  const cl_context_properties ctxProps[3] = { CL_CONTEXT_PLATFORM,
        (cl_context_properties)device.getPlatform()(), 0 };
  oclContext = clpp::Context(ctxProps, device);
  oclCmdQueue = clpp::CommandQueue(oclContext, oclDevice);
  
#ifdef HAVE_CLRX
  bool disableClrxAssembly = false;
  {
      const char* disaClrxAsmStr = ::getenv("ECLRXASM_DISABLE");
      disableClrxAssembly = (disaClrxAsmStr!=NULL && ::strcmp(disaClrxAsmStr, "1")==0);
  }
  if (!disableClrxAssembly)
    useClrxAssembly = prepareAssemblyOfClimbKernel();
#endif
  
  int wavefrontSize = 0;
#ifdef HAVE_CLRX
  if (!useClrxAssembly) // get wavefront size
#endif
    wavefrontSize = getWorkGroupSizeMultiple();
  
  oclProgram = clpp::Program(oclContext, (const char*)___enigma_cuda_lib_opencl_program_cl,
                    ___enigma_cuda_lib_opencl_program_cl_len);
  {
    char optionsBuf[150];
    size_t len = snprintf(optionsBuf, sizeof optionsBuf, "-DCIPHERTEXT_LEN=%d"
            " -DSCRAMBLER_STRIDE=%d", OpenCL_cipher_length, SCRAMBLER_STRIDE);
#ifdef HAVE_CLRX
    if (useClrxAssembly)
      strcat(optionsBuf, " -DWITHOUT_CLIMB_KERNEL=1");
    else // add wavefront size def
#endif
      snprintf(optionsBuf+len, (sizeof optionsBuf) - len,
               " -DWAVEFRONT_SIZE=%d", wavefrontSize);
    if (shortBigrams)
      strcat(optionsBuf, " -DSHORT_BIGRAMS=1");
    if (shortTrigrams)
      strcat(optionsBuf, " -DSHORT_TRIGRAMS=1");
    try
    { oclProgram.build(optionsBuf); }
    catch(const clpp::Error& error)
    {
      std::cerr << "BuildLogs:\n" << oclProgram.getBuildLog(oclDevice);
      throw;
    }
  }
  GenerateScramblerKernel = clpp::Kernel(oclProgram, "GenerateScramblerKernel");
#ifdef HAVE_CLRX
  if (!useClrxAssembly) // if not compiled in Assembler code
#endif
    ClimbKernel = clpp::Kernel(oclProgram, "ClimbKernel");
  FindBestResultKernel = clpp::Kernel(oclProgram, "FindBestResultKernel");
  FindBestResultKernel2 = clpp::Kernel(oclProgram, "FindBestResultKernel");
  
  GenerateScramblerKernelWGSize = GenerateScramblerKernel.getWorkGroupSize(oclDevice);
  GenerateScramblerKernelWGSize = std::min(GenerateScramblerKernelWGSize, size_t(64));
  thBlockShift = 1;
  int localShift = 6; // 64
  if (GenerateScramblerKernelWGSize < 64)
  {
    localShift = 5;
    GenerateScramblerKernelWGSize = 32;
    thBlockShift = 0;
  }
  GenerateScramblerKernel.setArg(2, cl_uint(thBlockShift));
  GenerateScramblerKernel.setArg(3, cl_uint(localShift));
  
  ClimbKernelWGSize = ClimbKernel.getWorkGroupSize(oclDevice);
  FindBestResultKernelWGSize = FindBestResultKernel.getWorkGroupSize(oclDevice);
  /// buffers
  d_ciphertextBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, MAX_MESSAGE_LENGTH);
  d_keyBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, sizeof(Key));
  d_wiringBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, sizeof(Wiring));
  d_orderBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, ALPSIZE);
  d_plugsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, ALPSIZE);
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
    d_fixedBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, ALPSIZE);
  d_unigramsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY,
                    ALPSIZE*sizeof(NGRAM_DATA_TYPE));
  d_bigramsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, ALPSIZE*ALPSIZE *
                  (shortBigrams ? sizeof(cl_ushort) : sizeof(NGRAM_DATA_TYPE)));
#ifdef DEBUG_CLIMB
  #if DEBUG_DUMP&1
  sregDumpBuffer = clpp::Buffer(oclContext, CL_MEM_WRITE_ONLY,
              sizeof(cl_uint)*ALPSIZE_TO3*3);
  #endif
  #if DEBUG_DUMP&2
  vregDumpBuffer = clpp::Buffer(oclContext, CL_MEM_WRITE_ONLY,
              sizeof(cl_uint)*ALPSIZE_TO3*CLRX_GroupSize*3);
  #endif
#endif
  
  GenerateScramblerKernel.setArgs(d_wiringBuffer, d_keyBuffer);
  
  ClimbKernel.setArgs(d_wiringBuffer, d_keyBuffer);
  ClimbKernel.setArg(7 - int(useClrxAssembly)*3, d_unigramsBuffer);
  ClimbKernel.setArg(8 - int(useClrxAssembly)*3, d_bigramsBuffer);
  ClimbKernel.setArg(9 - int(useClrxAssembly)*3, d_plugsBuffer);
  ClimbKernel.setArg(10 - int(useClrxAssembly)*3, d_orderBuffer);
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
    ClimbKernel.setArg(11, d_fixedBuffer);
  ClimbKernel.setArg(12 - int(useClrxAssembly)*3, d_ciphertextBuffer);
#ifdef DEBUG_CLIMB
  int argNo = 11;
  #if DEBUG_DUMP&1
  if (debugPart!=-1)
    ClimbKernel.setArg(argNo++, sregDumpBuffer);
  #endif
  #if DEBUG_DUMP&2
  if (debugPart!=-1)
    ClimbKernel.setArg(argNo++, vregDumpBuffer);
  #endif
#endif
  return true;
}

void CipherTextToDevice(string ciphertext_string)
{
  std::vector<int8_t> cipher = TextToNumbers(ciphertext_string);
  int8_t * cipher_data = cipher.data();
  oclCmdQueue.writeBuffer(d_ciphertextBuffer, 0, cipher.size(), cipher_data);
#ifdef DEBUG_CLIMB
  ::memcpy(d_ciphertext, cipher_data, cipher.size());
  task.count = (int)cipher.size();
#endif
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
    ClimbKernel.setArg(2, cl_uint(cipher.size()));
}

static std::unique_ptr<Unigrams> unigrams;
static std::unique_ptr<Bigrams> bigrams;
static std::unique_ptr<Trigrams> trigrams_obj;

void LoadNgrams(const string & uni_filename,
            const string & bi_filename, const string & tri_filename)
{
  if (uni_filename != "")
  {
    unigrams.reset(new Unigrams());
    unigrams->LoadFromFile(uni_filename);
  }
  if (bi_filename != "")
  {
    bigrams.reset(new Bigrams());
    shortBigrams = true;
    bigrams->LoadFromFile(bi_filename);
    for (int i = 0; i < ALPSIZE; i++)
      for (int j = 0; j < ALPSIZE; j++)
        if (bigrams->data[i][j]>0xffff || bigrams->data[i][j]<0)
        { shortBigrams = false; break; }
  }
  if (tri_filename != "")
  {
    trigrams_obj.reset(new Trigrams());
    shortTrigrams = true;
    trigrams_obj->LoadFromFile(tri_filename);
    for (int i = 0; i < ALPSIZE; i++)
      for (int j = 0; j < ALPSIZE; j++)
        for (int k = 0; k < ALPSIZE; k++)
        if (trigrams_obj->data[i][j][k]>0xffff || trigrams_obj->data[i][j][k]<0)
        { shortTrigrams = false; break; }
  }
  std::cerr << "Short bigrams: " << int(shortBigrams) <<
        ", Short trigrams: "<< int(shortTrigrams) << std::endl;
}

void NgramsToDevice()
{
  if (unigrams)
  {
    oclCmdQueue.writeBuffer(d_unigramsBuffer, 0,
                    sizeof(NGRAM_DATA_TYPE)*ALPSIZE, unigrams->data);
#ifdef DEBUG_CLIMB
    ::memcpy(d_unigrams, unigrams->data, sizeof(d_unigrams));
#endif
  }

  if (bigrams)
  {
    if (!shortBigrams)
      oclCmdQueue.writeBuffer(d_bigramsBuffer, 0,
                    sizeof(NGRAM_DATA_TYPE)*ALPSIZE*ALPSIZE, bigrams->data);
    else
    {
      clpp::BufferMapping mapping(oclCmdQueue, d_bigramsBuffer, true, CL_MAP_WRITE,
                      0, sizeof(cl_ushort)*ALPSIZE*ALPSIZE);
      cl_ushort* out = (cl_ushort*)mapping.get();
      const NGRAM_DATA_TYPE* in = (const NGRAM_DATA_TYPE*)bigrams->data;
      for (int i = 0; i < ALPSIZE*ALPSIZE; i++)
        out[i] = in[i];
    }
#ifdef DEBUG_CLIMB
    ::memcpy(d_bigrams, bigrams->data, sizeof(d_bigrams));
#endif
  }

  trigramsBufferPitch = 0;
  trigramsBuffer = clpp::Buffer();
  if (trigrams_obj)
  {
    //non-pitched array in device memory. slightly faster than pitched
    trigramsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY,
              (shortTrigrams? sizeof(cl_ushort) : sizeof(NGRAM_DATA_TYPE)) * ALPSIZE_TO3);
    trigramsBufferPitch =
        (shortTrigrams ? sizeof(cl_ushort) : sizeof(NGRAM_DATA_TYPE)) * ALPSIZE;
#ifdef DEBUG_CLIMB
    trigramsData = (NGRAM_DATA_TYPE*)malloc(sizeof(NGRAM_DATA_TYPE) * ALPSIZE_TO3);
    task.trigrams.data = (int8_t*)trigramsData;
    task.trigrams.pitch = sizeof(NGRAM_DATA_TYPE) * ALPSIZE;
    
    //data to device
    ::memcpy(trigramsData, trigrams_obj->data, sizeof(NGRAM_DATA_TYPE) * ALPSIZE_TO3);
#endif
    //data to device
    if (!shortTrigrams)
      oclCmdQueue.writeBuffer(trigramsBuffer, 0, sizeof(NGRAM_DATA_TYPE) * ALPSIZE_TO3,
                  trigrams_obj->data);
    else
    {
      clpp::BufferMapping mapping(oclCmdQueue, trigramsBuffer, true, CL_MAP_WRITE,
                      0, sizeof(cl_ushort)*ALPSIZE_TO3);
      cl_ushort* out = (cl_ushort*)mapping.get();
      const NGRAM_DATA_TYPE* in = (const NGRAM_DATA_TYPE*)trigrams_obj->data;
      for (int i = 0; i < ALPSIZE_TO3; i++)
        out[i] = in[i];
    }
  }
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
    ClimbKernel.setArg(5, cl_uint(trigramsBufferPitch));
  ClimbKernel.setArg(6 - 3*int(useClrxAssembly), trigramsBuffer);
  unigrams.reset();
  bigrams.reset();
  trigrams_obj.reset();
}

void OrderToDevice(const int8_t * order)
{
#ifdef DEBUG_CLIMB
  ::memcpy(d_order, order, ALPSIZE);
#endif
  oclCmdQueue.writeBuffer(d_orderBuffer, 0, ALPSIZE, order);
}

void PlugboardStringToDevice(string plugboard_string)
{
  Plugboard plugboard;
  plugboard.FromString(plugboard_string);
  PlugboardToDevice(plugboard);
}

void PlugboardToDevice(const Plugboard & plugboard)
{
#ifdef DEBUG_CLIMB
  ::memcpy(d_plugs, plugboard.plugs, ALPSIZE);
  ::memcpy(d_fixed, plugboard.fixed, sizeof(bool)*ALPSIZE);
#endif
  oclCmdQueue.enqueueWriteBuffer(d_plugsBuffer, 0, ALPSIZE, plugboard.plugs);
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
    oclCmdQueue.writeBuffer(d_fixedBuffer, 0, ALPSIZE, plugboard.fixed);
#ifdef HAVE_CLRX
  else
  {
    d_fixedValue = 0;
    for (int i = 0; i < ALPSIZE; i++)
      d_fixedValue |= plugboard.fixed[i] ? (1<<i) : 0;
    ClimbKernel.setArg(8, d_fixedValue);
  }
#endif
}

void SetUpResultsMemory(int count)
{
  resultsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_WRITE, count*sizeof(Result));
#ifdef DEBUG_CLIMB
  task.results = (Result*)malloc(count*sizeof(Result));
#endif
  FindBestResultKernel.setArg(0, resultsBuffer);
  FindBestResultKernel2.setArg(1, resultsBuffer);
  ClimbKernel.setArg(13 - 3*int(useClrxAssembly), resultsBuffer);
#ifdef DEBUG_CLIMB
  #if DEBUG_DUMP&1
  expectedSRegs = new cl_uint[ALPSIZE_TO3*3];
  #endif
  #if DEBUG_DUMP&2
  expectedVRegs = new cl_uint[ALPSIZE_TO3*CLRX_GroupSize*3];
  #endif
#endif
}

void InitializeArrays(const string cipher_string, int turnover_modes,        
  int score_kinds, int digits)
{
  //d_ciphertext
  CipherTextToDevice(cipher_string);
  //d_wiring
  SetUpScramblerMemory();
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
  {
    //allow_turnover
    ClimbKernel.setArg(14, cl_int(turnover_modes));
    //use unigrams
    ClimbKernel.setArg(15, cl_int(score_kinds));
  }

#ifdef DEBUG_CLIMB
  task.turnover_modes = turnover_modes;
  task.score_kinds = score_kinds;
#endif
  //d_results
  int count = (int)pow(ALPSIZE, digits);
  SetUpResultsMemory(count);
}


#ifdef DEBUG_CLIMB
/*
 * ClimbKernelHost
 * START
 */

int ComputeScramblerIndex(int char_pos, 
  const ScramblerStructure & stru,
  const RotorSettings & sett, const Wiring & wiring, int linear_idx)
{
  //retrieve notch info
  const int8_t * r_notch = wiring.notch_positions[stru.r_slot];
  const int8_t * m_notch = wiring.notch_positions[stru.m_slot];

  //period of the rotor turnovers
  int m_period = (r_notch[1] == NONE) ? ALPSIZE : HALF_ALPSIZE;
  int l_period = (m_notch[1] == NONE) ? ALPSIZE : HALF_ALPSIZE;
  l_period = --l_period * m_period;
  
  //current wheel position relative to the last notch
  int r_after_notch = sett.r_mesg - r_notch[0];
  if (r_after_notch < 0) r_after_notch += ALPSIZE;
  if (r_notch[1] != NONE && r_after_notch >= (r_notch[1] - r_notch[0]))
    r_after_notch -= r_notch[1] - r_notch[0];

  int m_after_notch = sett.m_mesg - m_notch[0];
  if (m_after_notch < 0) m_after_notch += ALPSIZE;
  if (m_notch[1] != NONE && m_after_notch >= (m_notch[1] - m_notch[0]))
    m_after_notch -= m_notch[1] - m_notch[0];
  
  //middle wheel turnover phase
  int m_phase = r_after_notch - 1;
  if (m_phase < 0) m_phase += m_period;
  
  //left wheel turnover phase
  int l_phase = m_phase - 1 + (m_after_notch - 1) * m_period;
  if (l_phase < 0) l_phase += l_period;

  //hacks
  if (m_after_notch == 0) l_phase += m_period;
  if (m_after_notch == 1 && r_after_notch == 1)
    l_phase -= l_period; //effectively sets l_phase to -1
  if (m_after_notch == 0 && r_after_notch == 0)
  {
    m_phase -= m_period;
    l_phase -= m_period;
    if (char_pos == 0) l_phase++;
  }

  //save debug info
  //	r_after_notch_display = r_after_notch;
  //	m_after_notch_display = m_after_notch;
  //	l_phase_display = l_phase;

  //number of turnovers
  int m_steps = (m_phase + char_pos + 1) / m_period;
  int l_steps = (l_phase + char_pos + 1) / l_period;

  //double step of the middle wheel
  m_steps += l_steps;
  
  //rotor core poistions to scrambling table index
  return mod26(sett.l_mesg - sett.l_ring + l_steps) * ALPSIZE_TO2 +
    mod26(sett.m_mesg - sett.m_ring + m_steps) * ALPSIZE +
    mod26(sett.r_mesg - sett.r_ring + char_pos + 1);
}

int ComputeScramblerIndexV(int char_pos, 
  const ScramblerStructure & stru,
  const RotorSettings & sett, const Wiring & wiring, int linear_idx)
{
  //retrieve notch info
  const int8_t * r_notch = wiring.notch_positions[stru.r_slot];
  const int8_t * m_notch = wiring.notch_positions[stru.m_slot];

  //period of the rotor turnovers
  int m_period = (r_notch[1] == NONE) ? ALPSIZE : HALF_ALPSIZE;
  int l_period = (m_notch[1] == NONE) ? ALPSIZE : HALF_ALPSIZE;
  l_period = --l_period * m_period;
  
  //current wheel position relative to the last notch
  int r_after_notch = sett.r_mesg - r_notch[0];
  if (r_after_notch < 0) r_after_notch += ALPSIZE;
  if (r_notch[1] != NONE && r_after_notch >= (r_notch[1] - r_notch[0]))
    r_after_notch -= r_notch[1] - r_notch[0];

  int m_after_notch = sett.m_mesg - m_notch[0];
  if (m_after_notch < 0) m_after_notch += ALPSIZE;
  if (m_notch[1] != NONE && m_after_notch >= (m_notch[1] - m_notch[0]))
    m_after_notch -= m_notch[1] - m_notch[0];
  
  //middle wheel turnover phase
  int m_phase = r_after_notch - 1;
  if (m_phase < 0) m_phase += m_period;
  
  //left wheel turnover phase
  int l_phase = m_phase - 1 + (m_after_notch - 1) * m_period;
  if (l_phase < 0) l_phase += l_period;

  //hacks
  if (m_after_notch == 0) l_phase += m_period;
  if (m_after_notch == 1 && r_after_notch == 1)
    l_phase -= l_period; //effectively sets l_phase to -1
  if (m_after_notch == 0 && r_after_notch == 0)
  {
    m_phase -= m_period;
    l_phase -= m_period;
    if (char_pos == 0) l_phase++;
  }

  //save debug info
  //	r_after_notch_display = r_after_notch;
  //	m_after_notch_display = m_after_notch;
  //	l_phase_display = l_phase;

  //number of turnovers
  int m_steps = (m_phase + char_pos + 1) / m_period;
  int l_steps = (l_phase + char_pos + 1) / l_period;

  //double step of the middle wheel
  m_steps += l_steps;
  
  //rotor core poistions to scrambling table index
  return mod26(sett.l_mesg - sett.l_ring + l_steps) * ALPSIZE_TO2 +
    mod26(sett.m_mesg - sett.m_ring + m_steps) * ALPSIZE +
    mod26(sett.r_mesg - sett.r_ring + char_pos + 1);
}

TurnoverLocation GetTurnoverLocation(const ScramblerStructure & stru,
  const RotorSettings sett, int ciphertext_length, const Wiring & wiring,
  int linear_idx )
{
  //rotors with two notches
  if (stru.r_slot > rotV && sett.r_ring >= HALF_ALPSIZE) 
      return toAfterMessage;
  if (stru.m_slot > rotV && sett.m_ring >= HALF_ALPSIZE) 
      return toAfterMessage;

  //does the left hand rotor turn right before the message?
  int8_t l_core_before = mod26(sett.l_mesg - sett.l_ring);
    int8_t l_core_first = ComputeScramblerIndex(0, stru, sett, wiring, linear_idx)
        / ALPSIZE_TO2;
  
  // DEBUG
#if DEBUG_DUMP
  if (debugPart==0) // ComputeScramblerIndex
  {
    expectedSRegs[linear_idx] = l_core_first;
    return toBeforeMessage;
  }
#endif
  // DEBUG
  if (l_core_first != l_core_before) return toBeforeMessage;

  //does it turn during the message?
    int8_t l_core_last = 
        ComputeScramblerIndex(ciphertext_length-1, stru, sett, wiring, linear_idx) 
        / ALPSIZE_TO2;
  if (l_core_last != l_core_first) return toDuringMessage;

  return toAfterMessage;
}

static int8_t shared_scrambling_table[20000];
const int8_t * ScramblerToShared(const int8_t * global_scrambling_table, dim3 threadIdx,
      int linear_idx)
{
  //global: ALPSIZE bytes at sequential addresses
  const int32_t * src = 
    reinterpret_cast<const int32_t *>(global_scrambling_table);

  //shared: same bytes in groups of 4 at a stride of 128
  int32_t * dst = reinterpret_cast<int32_t *>(shared_scrambling_table);

  //copy ALPSIZE bytes as 7 x 32-bit words
  /*int idx = (threadIdx.x & ~(SCRAMBLER_STRIDE-1)) * 7 +
              (threadIdx.x & (SCRAMBLER_STRIDE-1));
  for (int i = 0; i < 7; ++i) dst[idx + SCRAMBLER_STRIDE * i] = src[i];*/
  int idx = (threadIdx.x & ~31) * 7 + (threadIdx.x & 31);
  for (int i = 0; i < 7; ++i) dst[idx + 32 * i] = src[i];
  return &shared_scrambling_table[idx * 4];
}

int8_t Decode(const int8_t * plugboard, const int8_t * scrambling_table, dim3 threadIdx,
      int linear_idx)
{
  int8_t c = d_ciphertext[threadIdx.x];
  c = plugboard[c];
  //c = scrambling_table[(c & ~3) * SCRAMBLER_STRIDE + (c & 3)];
  c = scrambling_table[(c & ~3) * 32 + (c & 3)]; // orig
  c = plugboard[c];  
  return c;
}

void Sum(int count, volatile int * data, int * sum, size_t blockDim)
{
  int sum2 = 0;
  /*std::cout << "Sum of: ";
  for (size_t i = 0; i < count; i++)
    std::cout << " " << data[i];
  std::cout << "\n";*/
  for (size_t i = 0; i < count; i++)
    sum2 += data[i];
#if 0
  dim3 threadIdx = { 0, 0, 0 };
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  if ((threadIdx.x + 128) < count) data[threadIdx.x] += data[128 + threadIdx.x];
  }
  //__syncthreads();

  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  if (threadIdx.x < 64 && (threadIdx.x + 64) < count)
    data[threadIdx.x] += data[64 + threadIdx.x];
  }
  //__syncthreads();

  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  if (threadIdx.x < 32)
  {
    if ((threadIdx.x + 32) < count) data[threadIdx.x] += data[32 + threadIdx.x];
    if ((threadIdx.x + 16) < count) data[threadIdx.x] += data[16 + threadIdx.x];
    data[threadIdx.x] += data[8 + threadIdx.x];
    data[threadIdx.x] += data[4 + threadIdx.x];
    data[threadIdx.x] += data[2 + threadIdx.x];
    if (threadIdx.x == 0) *sum = data[0] + data[1];
  }
  }
  //__syncthreads();
#endif
    //std::cerr << "sum: " << *sum << "!=" << sum2 << "\n";
    *sum = sum2;
}

#define HISTO_SIZE 32

void IcScore(Block & block, const int8_t ** scrambling_table_tt, size_t blockDim,
             int linear_idx)
{
#if 0
  dim3 threadIdx = { 0, 0, 0 };
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  //init histogram
  if (threadIdx.x < HISTO_SIZE) block.score_buf[threadIdx.x] = 0;
  }
  //__syncthreads();
  
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
    const int8_t* scrambling_table = scrambling_table_tt[threadIdx.x];
  //compute histogram
  if (threadIdx.x < block.count)
  {
    int8_t c = Decode(block.plugs, scrambling_table, threadIdx);
    //atomicAdd((int *)&block.score_buf[c], 1);
    block.score_buf[c] += 1;
  }
  }
  //__syncthreads();

  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  //TODO: try lookup table here, ic[MAX_MESSAGE_LENGTH]
  if (threadIdx.x < HISTO_SIZE)
    block.score_buf[threadIdx.x] *= block.score_buf[threadIdx.x] - 1;

  //sum up
  if (threadIdx.x < HISTO_SIZE / 2)
  {
    block.score_buf[threadIdx.x] += block.score_buf[threadIdx.x + 16];
    block.score_buf[threadIdx.x] += block.score_buf[threadIdx.x + 8];
    block.score_buf[threadIdx.x] += block.score_buf[threadIdx.x + 4];
    block.score_buf[threadIdx.x] += block.score_buf[threadIdx.x + 2];
    if (threadIdx.x == 0) block.score = block.score_buf[0] + block.score_buf[1];
  }
  }

  //__syncthreads();
#endif
  for (int i = 0; i < HISTO_SIZE; i++)
    block.score_buf[i] = 0;
  for (int i = 0; i < block.count; i++)
  {
    const int8_t* scrambling_table = scrambling_table_tt[i];
    dim3 threadIdx{i, 0, 0};
    int8_t c = Decode(block.plugs, scrambling_table, threadIdx, linear_idx);
    block.score_buf[c] += 1;
  }
  for (int i = 0; i < HISTO_SIZE; i++)
    block.score_buf[i] *= block.score_buf[i] - 1;
  
  block.score = 0;
  for (int i = 0; i < HISTO_SIZE; i++)
    block.score += block.score_buf[i];
}


//TODO: put unigram table to shared memory
void UniScore(Block & block, const int8_t ** scrambling_table_tt, size_t blockDim,
              int linear_idx)
{
  dim3 threadIdx = { 0, 0, 0 };
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
    const int8_t* scrambling_table = scrambling_table_tt[threadIdx.x];
  if (threadIdx.x < block.count)
  {
    int8_t c = Decode(block.plugs, scrambling_table, threadIdx, linear_idx);
    block.score_buf[threadIdx.x] = block.unigrams[c];
  }
  }
  //__syncthreads();
  
  Sum(block.count, block.score_buf, &block.score, blockDim);
}

void BiScore(Block & block, const int8_t ** scrambling_table_tt, size_t blockDim,
             int linear_idx)
{
  dim3 threadIdx = { 0, 0, 0 };
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
    const int8_t* scrambling_table = scrambling_table_tt[threadIdx.x];
  if (threadIdx.x < block.count)
    block.plain_text[threadIdx.x] = Decode(block.plugs, scrambling_table, threadIdx,
                linear_idx);
  }
  //__syncthreads();

  //TODO: trigrams are faster than bigrams. 
  //is it because trigrams are not declared as constants?
  //or because their index is computed explicitly?
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  if (threadIdx.x < (block.count - 1))
    block.score_buf[threadIdx.x] = 
      d_bigrams[block.plain_text[threadIdx.x]]
               [block.plain_text[threadIdx.x + 1]];
  }
  //__syncthreads();

  Sum(block.count - 1, block.score_buf, &block.score, blockDim);
}


void TriScore(Block & block, const int8_t ** scrambling_table_tt, size_t blockDim,
              int linear_idx)
{
  //decode char
  dim3 threadIdx = { 0, 0, 0 };
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
    const int8_t* scrambling_table = scrambling_table_tt[threadIdx.x];
  if (threadIdx.x < block.count) 
    block.plain_text[threadIdx.x] = Decode(block.plugs, scrambling_table, threadIdx,
            linear_idx);
  }
  //__syncthreads();

  //look up scores
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  if (threadIdx.x < (block.count - 2))
    block.score_buf[threadIdx.x] = block.trigrams[
      block.plain_text[threadIdx.x] * ALPSIZE_TO2 +
      block.plain_text[threadIdx.x + 1] * ALPSIZE +
      block.plain_text[threadIdx.x+2]];
  }
  //__syncthreads();

  Sum(block.count - 2, block.score_buf, &block.score, blockDim);
}



void CalculateScore(Block & block, const int8_t ** scrambling_table_tt, size_t blockDim,
                  int linear_idx)
{
  switch (block.score_kind)
  {
  case skTrigram: TriScore(block, scrambling_table_tt, blockDim, linear_idx); break;
  case skBigram:  BiScore(block, scrambling_table_tt, blockDim, linear_idx); break;
  case skUnigram: UniScore(block, scrambling_table_tt, blockDim, linear_idx); break;
  case skIC:      IcScore(block, scrambling_table_tt, blockDim, linear_idx); break;
  }
}

int loopCount;

#define DUMP_IN_LOOPCOUNT 267

void TrySwap(int8_t i, int8_t k, const int8_t ** scrambling_table_tt, Block & block,
             size_t blockDim, int linear_idx)
{
   int old_score;
    int8_t x_tt[256], z_tt[256];
  old_score = block.score;

  if (d_fixed[i] || d_fixed[k])
    return;

  dim3 threadIdx = { 0, 0, 0 };
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  if (threadIdx.x == 0)
  {
    int8_t z, x;
    x = block.plugs[i];
    z = block.plugs[k];
    
    if (x == k)
    {
      block.plugs[i] = i;
      block.plugs[k] = k;
    }
    else
    {
      if (x != i)
      {
        block.plugs[i] = i;
        block.plugs[x] = x;
      };
      if (z != k)
      {
        block.plugs[k] = k;
        block.plugs[z] = z;
      };
      block.plugs[i] = k;
      block.plugs[k] = i;
    }
    
    x_tt[threadIdx.x] = x;
    z_tt[threadIdx.x] = z;
  }
  }
  //__syncthreads();

  CalculateScore(block, scrambling_table_tt, blockDim, linear_idx);

  //if (threadIdx.x == 0 && block.score <= old_score)
  if (block.score <= old_score)
  {
    block.score = old_score;

    int8_t x = x_tt[0], z = z_tt[0];
    block.plugs[z] = k;
    block.plugs[x] = i;
    block.plugs[k] = z;
    block.plugs[i] = x;
  }
  //__syncthreads();
}


void MaximizeScore(Block & block, const int8_t ** scrambling_table_tt, size_t blockDim,
                  int linear_idx)
{
  CalculateScore(block, scrambling_table_tt, blockDim, linear_idx);
  // DEBUG
  //expectedSRegs[linear_idx] = block.score;
  //if (block.score_kind == skTrigram)
    //return;
  loopCount = 0;
  // DEBUG
  for (int p = 0; p < ALPSIZE - 1; p++)
    for (int q = p + 1; q < ALPSIZE; q++)
    {
      TrySwap(d_order[p], d_order[q], scrambling_table_tt, block, blockDim, linear_idx);
      /*if (block.score_kind == skTrigram)
      {
        expectedSRegs[linear_idx] = block.score;
        if (loopCount == DUMP_IN_LOOPCOUNT)
          return;
        loopCount++;
      }*/
    }
  // DEBUG
#if DEBUG_DUMP
  if ((debugPart==1 && block.score_kind == skIC) ||
      (debugPart==2 && block.score_kind == skUnigram) ||
      (debugPart==3 && block.score_kind == skBigram) ||
      (debugPart==4 && block.score_kind == skTrigram))
    expectedSRegs[linear_idx] = block.score;
#endif
  // DEBUG
}

static int processedClimbTasks = 0;

static const int trigramItersHistCount = 16;
static int trigramItersHist[trigramItersHistCount];

void ClimbKernelHost(dim3 gridDim, size_t blockDim, dim3 blockIdx)
{
  Block block;
  RotorSettings sett;
  bool skip_this_key;
  Result * result;
  int linear_idx;
  
  block.score = 0;

  //if (threadIdx.x < ALPSIZE)
  for (int i = 0; i < ALPSIZE; i++)
  {
    block.plugs[i] = d_plugs[i];
    block.unigrams[i] = d_unigrams[i];
  }

  //if (threadIdx.x == 0)
  {
    block.trigrams = reinterpret_cast<int*>(task.trigrams.data);
    block.count = task.count;

    //ring and rotor settings to be tried
    sett.g_ring = 0;
    sett.l_ring = 0;

    //depending on the grid size, ring positions 
    //either from grid index or fixed (from d_key)
    sett.m_ring = (gridDim.y > ALPSIZE) ? blockIdx.y / ALPSIZE : d_key.sett.m_ring;
    sett.r_ring = (gridDim.y > 1) ? blockIdx.y % ALPSIZE : d_key.sett.r_ring;
    
    sett.g_mesg = d_key.sett.g_mesg;
    sett.l_mesg = (gridDim.x > ALPSIZE_TO2) ? blockIdx.x / ALPSIZE_TO2 : d_key.sett.l_mesg;
    sett.m_mesg = (gridDim.x > ALPSIZE) ? (blockIdx.x / ALPSIZE) % ALPSIZE : d_key.sett.m_mesg;
    sett.r_mesg = (gridDim.x > 1) ? blockIdx.x % ALPSIZE : d_key.sett.r_mesg;
    
    //element of results[] to store the output 
    linear_idx = blockIdx.z * ALPSIZE_TO2 + blockIdx.y * ALPSIZE + blockIdx.x;
    result = &task.results[linear_idx];
    result->index = linear_idx;
    result->score = -1;
    
    skip_this_key = ((gridDim.x > 1) &&
      (GetTurnoverLocation(d_key.stru, sett, block.count, d_wiring, linear_idx)
        & task.turnover_modes) == 0);
  }
  //__syncthreads();
  // DEBUG
  if (debugPart == 0)
    return;
  // DEBUG
  
  if (skip_this_key) return;
  
  processedClimbTasks++;
  
  dim3 threadIdx = { 0, 0, 0 };
  const int8_t * scrambling_table_tt[256];
  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
    const int8_t * scrambling_table;
    if (threadIdx.x < block.count)
      {
        scrambling_table = task.scrambler.data + 
        ComputeScramblerIndex(threadIdx.x, d_key.stru, sett, d_wiring, linear_idx) * 
          task.scrambler.pitch;
        scrambling_table = ScramblerToShared(scrambling_table, threadIdx, linear_idx);
      }
      scrambling_table_tt[threadIdx.x] = scrambling_table;
  }
  
  //IC once
  if (task.score_kinds & skIC)
  {
    block.score_kind = skIC;
    MaximizeScore(block, scrambling_table_tt, blockDim, linear_idx);
  }
  if (debugPart==1)
    return;
  
  //unigrams once
  if (task.score_kinds & skUnigram)
  {
    block.score_kind = skUnigram;
    MaximizeScore(block, scrambling_table_tt, blockDim, linear_idx);
  }
  if (debugPart==2)
    return;

  //bigrams once
  if (task.score_kinds & skBigram)
  {
    block.score_kind = skBigram;
    MaximizeScore(block, scrambling_table_tt, blockDim, linear_idx);
  }
  if (debugPart==3)
    return;

  //trigrams until convergence
  int iter = 0;
  if (task.score_kinds & skTrigram)
  {
    block.score_kind = skTrigram;
    block.score = 0;
    int old_score;
    do
    {
      old_score = block.score;
      MaximizeScore(block, scrambling_table_tt, blockDim, linear_idx);
      //return;
      trigramItersHist[std::min(iter, trigramItersHistCount-1)]++;
      iter++;
    } 
    while (block.score > old_score);
  }
  if (debugPart==4)
    return;
  

  for (threadIdx.x = 0; threadIdx.x < blockDim; threadIdx.x++)
  {
  //copy plugboard solution to global results array;
  if (threadIdx.x < ALPSIZE) result->plugs[threadIdx.x] = block.plugs[threadIdx.x];
  if (threadIdx.x == 0) result->score = block.score;
  }
}
/*
 * ClimbKernelHost stop
 */

static int callClimbNo = 0;

static int best_score = 0;

#define PER_CLIMB_CALL 1

#endif


Result Climb(int cipher_length, const Key & key, bool single_key)
{
#ifdef DEBUG_CLIMB
#if !defined(PER_CLIMB_CALL) || PER_CLIMB_CALL<=1
  bool doDebugClimb = true;
#else
  bool doDebugClimb = (callClimbNo % PER_CLIMB_CALL)==PER_CLIMB_CALL-1;
#endif
  if (doDebugClimb)
    std::cout << "\n\nCall climb no " << callClimbNo << std::endl;
#endif
  
  oclCmdQueue.enqueueWriteBuffer(d_keyBuffer, 0, sizeof(Key), &key);
#ifdef DEBUG_CLIMB
  if (doDebugClimb)
  {
  ::memset(shared_scrambling_table, 0, sizeof shared_scrambling_table);
  task.count = OpenCL_cipher_length;
  d_key = key;
  { // clear buffers
    clpp::BufferMapping mapping(oclCmdQueue, resultsBuffer, true, CL_MAP_WRITE, 0,
                        sizeof(Result)*ALPSIZE_TO3);
    ::memset(mapping.get(), 0, sizeof(Result)*ALPSIZE_TO3);
  }
  ::memset(task.results, 0, sizeof(Result)*ALPSIZE_TO3);
  #if DEBUG_DUMP&1
  { // sreg buffers
    clpp::BufferMapping mapping(oclCmdQueue, sregDumpBuffer, true, CL_MAP_WRITE, 0,
                        sizeof(cl_uint)*ALPSIZE_TO3*3);
    ::memset(mapping.get(), 0, sizeof(cl_uint)*ALPSIZE_TO3*3);
  }
  ::memset(expectedSRegs, 0, sizeof(cl_uint)*ALPSIZE_TO3*3);
  #endif
  #if DEBUG_DUMP&2
  { // vreg buffers
    clpp::BufferMapping mapping(oclCmdQueue, vregDumpBuffer, true, CL_MAP_WRITE, 0,
                        sizeof(cl_uint)*ALPSIZE_TO3*CLRX_GroupSize*3);
    ::memset(mapping.get(), 0, sizeof(cl_uint)*ALPSIZE_TO3*CLRX_GroupSize*3);
  }
  ::memset(expectedVRegs, 0, sizeof(cl_uint)*ALPSIZE_TO3*CLRX_GroupSize*3);
  #endif
  }
#endif
  int grid_size = single_key ? 1 : ALPSIZE_TO3;
  int block_size = std::max(32, cipher_length);
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
  {
    int shared_scrambler_size = ((cipher_length + (SCRAMBLER_STRIDE-1)) &
                    ~(SCRAMBLER_STRIDE-1)) * 28;
    ClimbKernel.setArg(16, clpp::Local(shared_scrambler_size));
  }

  int groupSize = !useClrxAssembly ? block_size : CLRX_GroupSize;
  clpp::Size3 workSize(grid_size*groupSize, 1, 1);
  clpp::Size3 localSize(groupSize, 1, 1);
#ifdef HAVE_CLRX
  if (!useClrxAssembly)
#endif
    oclCmdQueue.enqueueNDRangeKernel(ClimbKernel, workSize, localSize);
#ifdef HAVE_CLRX
  else // only one dimension for assembler version
    oclCmdQueue.enqueueNDRangeKernel(ClimbKernel, workSize[0], localSize[0]);
#endif
  
#ifdef DEBUG_CLIMB
  if (doDebugClimb)
  {
  // call for comparison
  dim3 blockDim { block_size, 1, 1 };
  dim3 gridDim { grid_size, 1, 1 };
  dim3 blockIdx = { 0, 0 , 0 };
  {
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++)
      for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++)
        for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++)
        {
          if ((blockIdx.x&4095)==0)
          {
            std::cout << "Climb blockIdx: " << blockIdx.x << "x" << blockIdx.y << "x" <<
                  blockIdx.z << "\r";
            std::cout.flush();
          }
          ClimbKernelHost(gridDim, blockDim.x, blockIdx);
        }
  }
  const int sgprElemsNum = 1;
  const int vgprElemsNum = 1;
  /* compare */
  bool error = false;
  #if DEBUG_DUMP&1
  if (debugPart!=-1)
  {
    clpp::BufferMapping mapping(oclCmdQueue, sregDumpBuffer, true, CL_MAP_READ, 0,
            sizeof(cl_uint)*ALPSIZE_TO3*3);
    const cl_uint* results = (const cl_uint*)mapping.get();
    for (int i = 0; i < ALPSIZE_TO3; i++)
      for (int j = 0; j < sgprElemsNum; j++)
        if (expectedSRegs[i*sgprElemsNum + j] != results[i*sgprElemsNum + j])
        {
          std::cerr << "SRegArray[" << i << "][" << j << "] mismatch: " <<
                int(expectedSRegs[i*sgprElemsNum + j]) << "!=" <<
                int(results[i*sgprElemsNum + j]) << "\n";
          error = true;
        }
  }
  #endif
  #if DEBUG_DUMP&2
  if (debugPart!=-1)
  {
    clpp::BufferMapping mapping(oclCmdQueue, vregDumpBuffer, true, CL_MAP_READ, 0,
            sizeof(cl_uint)*ALPSIZE_TO3*CLRX_GroupSize*3);
    const cl_uint* results = (const cl_uint*)mapping.get();
    for (int i = 0; i < ALPSIZE_TO3; i++)
      for (int j = 0; j < CLRX_GroupSize; j++)
        for (int k = 0; k < vgprElemsNum; k++)
          if (expectedVRegs[(i*CLRX_GroupSize+j)*vgprElemsNum + k] !=
                results[(i*CLRX_GroupSize+j)*vgprElemsNum + k])
          {
            std::cerr << "VRegArray[" << i << "][" << j << "][" << k << "] mismatch: " <<
                  int(expectedVRegs[(i*CLRX_GroupSize+j)*vgprElemsNum + k]) << "!=" <<
                  int(results[(i*CLRX_GroupSize+j)*vgprElemsNum + k]) << "\n";
            error = true;
          }
  }
  #endif
  #if 0
  {
    clpp::BufferMapping mapping(oclCmdQueue, resultsBuffer, true, CL_MAP_READ, 0,
                                sizeof(Result)*gridDim.x*gridDim.y*gridDim.z);
    const Result* resResults = (const Result*)mapping.get();
    for (size_t i = 0; i < gridDim.x*gridDim.y*gridDim.z; i++)
    {
      const Result& resResult = resResults[i];
      if (expectedSRegs[i]!=1)
      {
        if (resResult.score != -1 || resResult.index != i)
        {
          std::cerr << "ResResult[" << i << "] failed: " <<
              resResult.score << ", " << resResult.index << "\n";
          error = true;
        }
      }
      else
      {
        if (resResult.score != 0 || resResult.index != 0)
        {
          std::cerr << "ResResult[" << i << "] failed: " <<
              resResult.score << ", " << resResult.index << "\n";
          error = true;
        }
      }
    }
  }
  #endif
  #ifdef DEBUG_RESULTS
  if (debugPart==-1)
  {
    std::cout << "\nChecking results" << std::endl;
    clpp::BufferMapping mapping(oclCmdQueue, resultsBuffer, true, CL_MAP_READ, 0,
                                sizeof(Result)*gridDim.x*gridDim.y*gridDim.z);
    const Result* resResults = (const Result*)mapping.get();
    for (size_t i = 0; i < gridDim.x*gridDim.y*gridDim.z; i++)
    {
      const Result& exResult = task.results[i];
      const Result& resResult = resResults[i];
      if (exResult.score != resResult.score)
      {
        std::cerr << "Result " << i << " score not match: " << exResult.score << "!=" <<
              resResult.score << "\n";
        error = true;
      }
      if (exResult.index != resResult.index)
      {
        std::cerr << "Result " << i << " index not match: " << exResult.index << "!=" <<
              resResult.index << "\n";
        error = true;
      }
      if (exResult.score < 0)
      {
        //std::cerr << "Skipped: " << i << "\n";
        continue;
      }
      for (int j = 0; j < ALPSIZE; j++)
        if (exResult.plugs[j] != resResult.plugs[j])
        {
          std::cerr << "Result " << i << " plug[" << j << "] not match: " <<
              int(exResult.plugs[j]) << "!=" << int(resResult.plugs[j]) << "\n";
              error = true;
        }
    }
  }
  #endif
  if (error)
  {
    std::cerr << "Have some errors in Climb call " << callClimbNo << "!" << std::endl;
    std::cerr << "key = Key{\n  { EnigmaModel(" << int(key.stru.model) << "),\n"
                 "    ReflectorType(" << int(key.stru.ukwnum) << "),\n"
                 "    RotorType(" << int(key.stru.g_slot) << "),\n"
                 "    RotorType(" << int(key.stru.l_slot) << "),\n"
                 "    RotorType(" << int(key.stru.m_slot) << "),\n"
                 "    RotorType(" << int(key.stru.r_slot) << ") },\n"
                 "  { " << key.sett.g_ring << ", " << key.sett.l_ring << ", " <<
                 key.sett.m_ring << ", " << key.sett.r_ring << ",\n"
                 "    " << key.sett.g_mesg << ", " << key.sett.l_mesg << ", " <<
                 key.sett.m_mesg << ", " << key.sett.r_mesg << " } };\n";
          //int8_t reflectors[REFLECTOR_TYPE_CNT][ALPSIZE];
	//int8_t rotors[ROTOR_TYPE_CNT][ALPSIZE];
	//int8_t reverse_rotors[ROTOR_TYPE_CNT][ALPSIZE];
	//int8_t notch_positions[ROTOR_TYPE_CNT][2];
    std::cerr << "wiring = {\n  {\n";
    for (int i = 0; i < REFLECTOR_TYPE_CNT; i++)
    {
      std::cerr << "    { ";
      for (int j = 0; j < ALPSIZE; j++)
        std::cerr << int(wiring.reflectors[i][j]) << (j+1<ALPSIZE ? ", ": " }");
      std::cerr << (i+1<REFLECTOR_TYPE_CNT ? ",\n" : " },\n");
    }
    std::cerr << "  {\n";
    for (int i = 0; i < ROTOR_TYPE_CNT; i++)
    {
      std::cerr << "    { ";
      for (int j = 0; j < ALPSIZE; j++)
        std::cerr << int(wiring.rotors[i][j]) << (j+1<ALPSIZE ? ", ": " }");
      std::cerr << (i+1<ROTOR_TYPE_CNT ? ",\n" : " },\n");
    }
    std::cerr << "  {\n";
    for (int i = 0; i < ROTOR_TYPE_CNT; i++)
    {
      std::cerr << "    { ";
      for (int j = 0; j < ALPSIZE; j++)
        std::cerr << int(wiring.reverse_rotors[i][j]) << (j+1<ALPSIZE ? ", ": " }");
      std::cerr << (i+1<ROTOR_TYPE_CNT ? ",\n" : " },\n");
    }
    std::cerr << "  {\n";
    for (int i = 0; i < ROTOR_TYPE_CNT; i++)
    {
      std::cerr << "    { ";
      for (int j = 0; j < 2; j++)
        std::cerr << int(wiring.notch_positions[i][j]) << (j+1<2 ? ", ": " }");
      std::cerr << (i+1<ROTOR_TYPE_CNT ? ",\n" : " }\n");
    }
    std::cerr << " }\n};\n";
    std::cerr << "scramblerData = {\n";
    for (int i = 0; i < ALPSIZE_TO3; i++)
    {
      for (int j = 0; j < ALPSIZE; j++)
        std::cerr << int(task.scrambler.data[task.scrambler.pitch*i + j]) <<
            (j+1<ALPSIZE ? ", " : ",\n");
    }
    std::cerr << "};\n";
    if (OpenCL_score_kinds & skUnigram)
    {
      std::cerr << "unigrams = { ";
      for (int i = 0; i < ALPSIZE; i++)
        std::cerr << d_unigrams[i] << ((i&7)!=7 ? ", " : ",\n");
      std::cerr << "};\n";
    }
    if (OpenCL_score_kinds & skBigram)
    {
      std::cerr << "bigrams = { ";
      for (int i = 0; i < ALPSIZE_TO2; i++)
        std::cerr << d_bigrams[i] << ((i&7)!=7 ? ", " : ",\n");
      std::cerr << "};\n";
    }
    if (OpenCL_score_kinds & skTrigram)
    {
      std::cerr << "trigrams = { ";
      const int* trigs = (const int*)task.trigrams.data;
      for (int i = 0; i < ALPSIZE_TO3; i++)
        std::cerr << trigs[i] << ((i&7)!=7 ? ", " : ",\n");
      std::cerr << "};\n";
    }
    std::cerr << "d_plugs = { ";
    for (int i = 0; i < ALPSIZE; i++)
      std::cerr << int(d_plugs[i]) << ((i&31)!=31 ? ", " : ",\n");
    std::cerr << "};\n";
    std::cerr << "d_order = { ";
    for (int i = 0; i < ALPSIZE; i++)
      std::cerr << int(d_order[i]) << ((i&15)!=15 ? ", " : ",\n");
    std::cerr << "};\n";
    std::cerr << "d_fixed = { ";
    for (int i = 0; i < ALPSIZE; i++)
      std::cerr << int(d_fixed[i]) << ((i&15)!=15 ? ", " : ",\n");
    std::cerr << "};\n";
    std::cerr << "d_ciphertext = { ";
    for (int i = 0; i < OpenCL_cipher_length; i++)
      std::cerr << int(d_ciphertext[i]) << ((i&15)!=15 ? ", " : ",\n");
    std::cerr << "};\n";
    std::cerr << "score_kinds = " << int(OpenCL_score_kinds) << ";\n";
    std::cerr << "turnover_modes = " << int(OpenCL_turnover_modes) << ";\n";
    std::cerr.flush();
    ::exit(1);
  }
  callClimbNo++;
  Result ret = GetBestResult(ALPSIZE_TO3);
  best_score = std::max(ret.score, best_score);
  std::cout << "Best score: " << best_score << std::endl;
  if (debugPart!=-1)
    ::exit(0); // stop when is not full test
  // allow next call if full test
  return ret;
  }
  else
  {
    Result ret = GetBestResult(ALPSIZE_TO3);
    callClimbNo++;
    best_score = std::max(ret.score, best_score);
    return ret;
  }
#else
  return GetBestResult(ALPSIZE_TO3);
#endif
}

unsigned int nextPow2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void ComputeDimensions(int count, int & grid_size, int & block_size)
{
  block_size = (count < REDUCE_MAX_THREADS * 2) ?
        nextPow2((count + 1) / 2) : REDUCE_MAX_THREADS;
  grid_size = (count + (block_size * 2 - 1)) / (block_size * 2);
}

Result GetBestResult(int count)
{
  int grid_size, block_size;
  ComputeDimensions(count, grid_size, block_size);
  
  if (d_tempSize < grid_size*sizeof(Result) || d_tempBuffer()==NULL)
  {
    d_tempBuffer = clpp::Buffer();
    d_tempBuffer = clpp::Buffer(oclContext, CL_MEM_READ_WRITE, grid_size * sizeof(Result));
    d_tempSize = grid_size*sizeof(Result);
    FindBestResultKernel.setArg(1, d_tempBuffer);
    FindBestResultKernel2.setArg(0, d_tempBuffer);
  }
  
  FindBestResultKernel.setArg(2, cl_uint(count));
  oclCmdQueue.enqueueNDRangeKernel(FindBestResultKernel,
                            grid_size*block_size, block_size);
  
  bool swapped = true;
  int s = grid_size;
  while (s > 1)
  {
    ComputeDimensions(s, grid_size, block_size);
    const clpp::Kernel& kernel = swapped ? FindBestResultKernel2 : FindBestResultKernel;
    kernel.setArg(2, cl_uint(s));
    oclCmdQueue.enqueueNDRangeKernel(kernel, grid_size*block_size, block_size);
    s = (s + (block_size * 2 - 1)) / (block_size * 2);
    swapped = !swapped;
  }
  Result result;
  oclCmdQueue.readBuffer(swapped ? d_tempBuffer : resultsBuffer,
                         0, sizeof(Result), &result);
  return result;
}


string DecodeMessage(const string & ciphertext,
  const string & key_string, const string & plugboard_string)
{
  Plugboard plugboard;
  plugboard.FromString(plugboard_string);

  return DecodeMessage(ciphertext, key_string, plugboard.plugs);
}

string DecodeMessage(const string & ciphertext,
  const string & key_string, const int8_t * plugs)
{
  Key key;
  key.FromString(key_string);

  string result = ciphertext;

  for (size_t i = 0; i < result.length(); i++)
  {
    key.Step();
    result[i] = ToChar(DecodeLetter(ToNum(result[i]), key, plugs));
  }

  return LowerCase(result);
}

int8_t DecodeLetter(int8_t c, const Key & key, const int8_t * plugs)
{
  int8_t r = mod26(key.sett.r_mesg - key.sett.r_ring);
  int8_t m = mod26(key.sett.m_mesg - key.sett.m_ring);
  int8_t l = mod26(key.sett.l_mesg - key.sett.l_ring);
  int8_t g = mod26(key.sett.g_mesg - key.sett.g_ring);

  c = plugs[c];

  c = wiring.rotors[key.stru.r_slot][mod26(c + r)] - r;
  c = wiring.rotors[key.stru.m_slot][mod26(c + m)] - m;
  c = wiring.rotors[key.stru.l_slot][mod26(c + l)] - l;
  c = wiring.rotors[key.stru.g_slot][mod26(c + g)] - g;

  c = wiring.reflectors[key.stru.ukwnum][mod26(c)];

  c = wiring.reverse_rotors[key.stru.g_slot][mod26(c + g)] - g;
  c = wiring.reverse_rotors[key.stru.l_slot][mod26(c + l)] - l;
  c = wiring.reverse_rotors[key.stru.m_slot][mod26(c + m)] - m;
  c = wiring.reverse_rotors[key.stru.r_slot][mod26(c + r)] - r;

  return plugs[mod26(c)];
}

void CleanUpGPU()
{
  FindBestResultKernel = clpp::Kernel();
  FindBestResultKernel2 = clpp::Kernel();
  ClimbKernel = clpp::Kernel();
  GenerateScramblerKernel = clpp::Kernel();
  
  d_ciphertextBuffer = clpp::Buffer();
  d_wiringBuffer = clpp::Buffer();
  d_keyBuffer = clpp::Buffer();
  scramblerDataBuffer = clpp::Buffer();
  d_orderBuffer = clpp::Buffer();
  d_plugsBuffer = clpp::Buffer();
  d_fixedBuffer = clpp::Buffer();
  d_tempBuffer = clpp::Buffer();
  d_unigramsBuffer = clpp::Buffer();
  d_bigramsBuffer = clpp::Buffer();
  resultsBuffer = clpp::Buffer();
  trigramsBuffer = clpp::Buffer();
#ifdef DEBUG_CLIMB
  #if DEBUG_DUMP&2
  vregDumpBuffer = clpp::Buffer();
  #endif
  #if DEBUG_DUMP&1
  sregDumpBuffer = clpp::Buffer();
  #endif
#endif
  
  oclCmdQueue = clpp::CommandQueue();
  oclProgram = clpp::Program();
  oclClrxProgram = clpp::Program();
  oclContext = clpp::Context();
  
  unigrams.reset();
  bigrams.reset();
  trigrams_obj.reset();
}
