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
#include <cmath>
#include <memory>
#include <clpp.h>
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

#define ACCEPT_ONLY_PREFERRED_PLATFORM 1
// for AMD
#define PLATFORM_VENDOR "Advanced Micro Devices, Inc."
// for NVIDIA
//#define PLATFORM_VENDOR "NVIDIA Corporation"

static clpp::Device oclDevice;
static clpp::Context oclContext;
static clpp::Program oclProgram;
static clpp::CommandQueue oclCmdQueue;
static clpp::Kernel GenerateScramblerKernel;
static size_t GenerateScramblerKernelWGSize;
static clpp::Kernel ClimbKernel;
static size_t ClimbKernelWGSize;
static clpp::Kernel FindBestResultKernel;
static clpp::Kernel FindBestResultKernel2;
static size_t FindBestResultKernelWGSize;
static cl_uint thBlockShift = 0;

// buffers
static clpp::Buffer d_ciphertextBuffer;
static clpp::Buffer d_wiringBuffer;
static clpp::Buffer d_keyBuffer;
static size_t scramblerDataPitch;
static clpp::Buffer scramblerDataBuffer;
static clpp::Buffer d_orderBuffer;
static clpp::Buffer d_plugsBuffer;
static clpp::Buffer d_fixedBuffer;
static clpp::Buffer d_tempBuffer;
static clpp::Buffer d_unigramsBuffer;
static clpp::Buffer d_bigramsBuffer;
static size_t d_tempSize = 0;
static clpp::Buffer resultsBuffer;
static size_t trigramsBufferPitch;
static clpp::Buffer trigramsBuffer;

extern "C"
{

int8_t mod26(const int16_t x)
{
  return (ALPSIZE * 2 + x) % ALPSIZE;
}

};


void SetUpScramblerMemory()
{
  oclCmdQueue.writeBuffer(d_wiringBuffer, 0, sizeof(Wiring), &wiring);
  scramblerDataPitch = (28 + 15) & ~size_t(15);
  scramblerDataBuffer = clpp::Buffer(oclContext, CL_MEM_READ_WRITE,
                  scramblerDataPitch*ALPSIZE_TO3);
  GenerateScramblerKernel.setArg(4, cl_uint(scramblerDataPitch));
  GenerateScramblerKernel.setArg(5, scramblerDataBuffer);
  ClimbKernel.setArg(3, cl_uint(scramblerDataPitch));
  ClimbKernel.setArg(4, scramblerDataBuffer);
}

void GenerateScrambler(const Key & key)
{
  oclCmdQueue.enqueueWriteBuffer(d_keyBuffer, 0, sizeof(Key), &key);
  clpp::Size3  dimGrid(ALPSIZE>>thBlockShift, ALPSIZE, ALPSIZE);
  clpp::Size3 dimBlock(GenerateScramblerKernelWGSize, 1, 1);
  dimGrid[0] *= dimBlock[0];
  oclCmdQueue.enqueueNDRangeKernel(GenerateScramblerKernel, dimGrid, dimBlock).wait();
}

std::string trimSpaces(const std::string& s)
{
    std::string::size_type pos = s.find_first_not_of(" \n\t\r\v\f");
    if (pos == std::string::npos)
        return "";
    std::string::size_type endPos = s.find_last_not_of(" \n\t\r\v\f");
    return s.substr(pos, endPos+1-pos);
}

/*
 * OpenCL init
 */
bool SelectGpuDevice(int req_major, int req_minor, int settings_device,
                bool silent, int ciphertext_length)
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
  
  const char* cldevStr = getenv("CLDEV");
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
  
  /*
   * creating opencl stuff
   * building program and creating kernels
   */
  const cl_context_properties ctxProps[3] = { CL_CONTEXT_PLATFORM,
        (cl_context_properties)device.getPlatform()(), 0 };
  oclContext = clpp::Context(ctxProps, device);
  oclCmdQueue = clpp::CommandQueue(oclContext, oclDevice);
  oclProgram = clpp::Program(oclContext, (const char*)___enigma_cuda_lib_opencl_program_cl,
                    ___enigma_cuda_lib_opencl_program_cl_len);
  {
    char optionsBuf[64];
    snprintf(optionsBuf, sizeof optionsBuf, "-DCIPHERTEXT_LEN=%d"
            " -DSCRAMBLER_STRIDE=%d", ciphertext_length, SCRAMBLER_STRIDE);
    oclProgram.build(optionsBuf);
  }
  GenerateScramblerKernel = clpp::Kernel(oclProgram, "GenerateScramblerKernel");
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
  d_fixedBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, ALPSIZE);
  d_unigramsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY,
                    ALPSIZE*sizeof(NGRAM_DATA_TYPE));
  d_bigramsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY,
                    ALPSIZE*ALPSIZE*sizeof(NGRAM_DATA_TYPE));
  
  GenerateScramblerKernel.setArgs(d_wiringBuffer, d_keyBuffer);
  
  ClimbKernel.setArgs(d_wiringBuffer, d_keyBuffer);
  ClimbKernel.setArg(7, d_unigramsBuffer);
  ClimbKernel.setArg(8, d_bigramsBuffer);
  ClimbKernel.setArg(9, d_plugsBuffer);
  ClimbKernel.setArg(10, d_orderBuffer);
  ClimbKernel.setArg(11, d_fixedBuffer);
  ClimbKernel.setArg(12, d_ciphertextBuffer);
  return true;
}

void CipherTextToDevice(string ciphertext_string)
{
  std::vector<int8_t> cipher = TextToNumbers(ciphertext_string);
  int8_t * cipher_data = cipher.data();
  oclCmdQueue.writeBuffer(d_ciphertextBuffer, 0, cipher.size(), cipher_data);
  ClimbKernel.setArg(2, cl_uint(cipher.size()));
}

void NgramsToDevice(const string & uni_filename,        
            const string & bi_filename, const string & tri_filename)
{
  if (uni_filename != "")
  {
    Unigrams unigrams;
    unigrams.LoadFromFile(uni_filename);
    oclCmdQueue.writeBuffer(d_unigramsBuffer, 0,
                    sizeof(NGRAM_DATA_TYPE)*ALPSIZE, unigrams.data);
  }

  if (bi_filename != "")
  {
    Bigrams bigrams;
    bigrams.LoadFromFile(bi_filename);
    oclCmdQueue.writeBuffer(d_bigramsBuffer, 0,
                    sizeof(NGRAM_DATA_TYPE)*ALPSIZE*ALPSIZE, bigrams.data);
  }

  trigramsBufferPitch = 0;
  trigramsBuffer = clpp::Buffer();
  if (tri_filename != "")
  {
    //trigram data
    Trigrams trigrams_obj;
    trigrams_obj.LoadFromFile(tri_filename);

    //non-pitched array in device memory. slightly faster than pitched
    trigramsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY,
                    sizeof(NGRAM_DATA_TYPE) * ALPSIZE_TO3);
    trigramsBufferPitch = sizeof(NGRAM_DATA_TYPE) * ALPSIZE;

    //data to device
    oclCmdQueue.writeBuffer(trigramsBuffer, 0, sizeof(NGRAM_DATA_TYPE) * ALPSIZE_TO3,
                  trigrams_obj.data);
  }
  ClimbKernel.setArg(5, cl_uint(trigramsBufferPitch));
  ClimbKernel.setArg(6, trigramsBuffer);
}

void OrderToDevice(const int8_t * order)
{
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
  oclCmdQueue.enqueueWriteBuffer(d_plugsBuffer, 0, ALPSIZE, plugboard.plugs);
  oclCmdQueue.writeBuffer(d_fixedBuffer, 0, ALPSIZE, plugboard.fixed);
}

void SetUpResultsMemory(int count)
{
  resultsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_WRITE, count*sizeof(Result));
  FindBestResultKernel.setArg(0, resultsBuffer);
  FindBestResultKernel2.setArg(1, resultsBuffer);
  ClimbKernel.setArg(13, resultsBuffer);
}

void InitializeArrays(const string cipher_string, int turnover_modes,        
  int score_kinds, int digits)
{
  //d_ciphertext
  CipherTextToDevice(cipher_string);
  //d_wiring
  SetUpScramblerMemory();
  //allow_turnover
  ClimbKernel.setArg(14, cl_int(turnover_modes));
  //use unigrams
  ClimbKernel.setArg(15, cl_int(score_kinds));

  //d_results
  int count = (int)pow(ALPSIZE, digits);
  SetUpResultsMemory(count);
}

Result Climb(int cipher_length, const Key & key, bool single_key)
{
  oclCmdQueue.enqueueWriteBuffer(d_keyBuffer, 0, sizeof(Key), &key);
  int grid_size = single_key ? 1 : ALPSIZE_TO3;
  int block_size = std::max(32, cipher_length);
  int shared_scrambler_size = ((cipher_length + (SCRAMBLER_STRIDE-1)) &
                    ~(SCRAMBLER_STRIDE-1)) * 28;
  ClimbKernel.setArg(16, clpp::Local(shared_scrambler_size));
  clpp::Size3 workSize(grid_size*block_size, 1, 1);
  clpp::Size3 localSize(block_size, 1, 1);
  oclCmdQueue.enqueueNDRangeKernel(ClimbKernel, workSize, localSize);
  return GetBestResult(ALPSIZE_TO3);
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
  
  oclCmdQueue = clpp::CommandQueue();
  oclProgram = clpp::Program();
  oclContext = clpp::Context();
}
