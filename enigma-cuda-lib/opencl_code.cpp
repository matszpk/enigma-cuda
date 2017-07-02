/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */
/* Copyright (c) 2017 Mateusz Szpakowski                               */

#define __CLPP_CL_ABI_VERSION 101
#define __CLPP_CL_VERSION 101

#include <stdexcept>
#include <iostream>
#include <cerrno>
#include <cstdlib>
#include <clpp.h>
#include <opencl_code.h>

#define REDUCE_MAX_THREADS 256

static clpp::Device oclDevice;
static clpp::Context oclContext;
static clpp::Program oclProgram;
static clpp::CommandQueue oclCmdQueue;
static clpp::Kernel GenerateScramblerKernel;
static size_t GenerateScramblerKernelWGSize;
static clpp::Kernel ClimbKernel;
static size_t ClimbKernelWGSize;
static clpp::Kernel FindBestResultKernel;
static size_t FindBestResultKernelWGSize;
static cl_uint thBlockShift = 0;

// buffers
static clpp::Buffer d_ciphertextBuffer;
static clpp::Buffer d_wiringBuffer;
static clpp::Buffer d_keyBuffer;
static clpp::Buffer scramblerDataPitchBuffer;
static clpp::Buffer d_orderBuffer;
static clpp::Buffer d_plugsBuffer;
static clpp::Buffer d_fixedBuffer;
static clpp::Buffer d_tempBuffer;
static size_t d_tempSize = 0;
static clpp::Buffer resultsBuffer;

Result * d_temp;

Task h_task;

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
  h_task.scrambler.pitch = (h_task.scrambler.pitch + 15) & ~size_t(16);
  scramblerDataPitchBuffer = clpp::Buffer(oclContext, CL_MEM_READ_WRITE,
                  h_task.scrambler.pitch*ALPSIZE_TO3);
  GenerateScramblerKernel.setArg(3, h_task.scrambler.pitch);
  GenerateScramblerKernel.setArg(4, scramblerDataPitchBuffer);
}

void GenerateScrambler(const Key & key)
{
  oclCmdQueue.writeBuffer(d_keyBuffer, 0, sizeof(Key), &key);
  clpp::Size3  dimGrid(ALPSIZE>>thBlockShift, ALPSIZE, ALPSIZE);
  size_t dimBlock = GenerateScramblerKernelWGSize;
  dimGrid[0] *= dimBlock;
  dimGrid[1] *= dimBlock;
  dimGrid[2] *= dimBlock;
  oclCmdQueue.enqueueNDRangeKernel(GenerateScramblerKernel, dimGrid, dimBlock).wait();
}

/*
 * OpenCL init
 */
bool SelectGpuDevice(int req_major, int req_minor, bool silent)
{
  std::vector<clpp::Platform> platforms = clpp::Platform::get();
  if (platforms.empty())
  {
    std::cerr << "OpenCL platform not found. Terminating...";
    return false;
  }
  int best_device = 0;
  const std::vector<clpp::Device> devices = platforms[0].getGPUDevices();
  switch(devices.size())
  {
    case 0:
      std::cerr << "OpenCL platform not found. Terminating...";
      return false;
    case 1:
      break;
    default:
    {
      const char* cldevStr = getenv("CLDEV");
      if (cldevStr != nullptr)
        best_device = atoi(cldevStr);
      else // just select
      {
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
      break;
    }
  }
  const clpp::Device& device = devices[best_device];
  if (!silent)
    std::cout << "Found GPU '" << best_device << ": " << device.getName() << std::endl;
  oclDevice = device;
  
  /*
   * creating opencl stuff
   * building program and creating kernels
   */
  oclContext = clpp::Context(device);
  oclCmdQueue = clpp::CommandQueue(oclContext, oclDevice);
  oclProgram = clpp::Program(oclContext, (const char*)___enigma_cuda_lib_opencl_program_cl,
                    ___enigma_cuda_lib_opencl_program_cl_len);
  oclProgram.build();
  GenerateScramblerKernel = clpp::Kernel(oclProgram, "GenerateScramblerKernel");
  ClimbKernel = clpp::Kernel(oclProgram, "ClimbKernel");
  FindBestResultKernel = clpp::Kernel(oclProgram, "FindBestResultKernel");
  
  GenerateScramblerKernel.setArgs(d_wiringBuffer, d_keyBuffer, cl_uint(0), cl_uint(0));
  
  GenerateScramblerKernelWGSize = GenerateScramblerKernel.getWorkGroupSize(oclDevice);
  GenerateScramblerKernelWGSize = std::min(GenerateScramblerKernelWGSize, size_t(64));
  thBlockShift = 1;
  if (GenerateScramblerKernelWGSize < 64)
  {
    GenerateScramblerKernelWGSize = 32;
    thBlockShift = 0;
  }
    
  ClimbKernelWGSize = ClimbKernel.getWorkGroupSize(oclDevice);
  FindBestResultKernelWGSize = FindBestResultKernel.getWorkGroupSize(oclDevice);
  /// buffers
  d_ciphertextBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, MAX_MESSAGE_LENGTH);
  d_keyBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, sizeof(Key));
  d_wiringBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, sizeof(Wiring));
  d_orderBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, ALPSIZE);
  d_plugsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, ALPSIZE);
  d_fixedBuffer = clpp::Buffer(oclContext, CL_MEM_READ_ONLY, ALPSIZE);
  return true;
}

void CipherTextToDevice(string ciphertext_string)
{
  std::vector<int8_t> cipher = TextToNumbers(ciphertext_string);
  int8_t * cipher_data = cipher.data();
  oclCmdQueue.writeBuffer(d_ciphertextBuffer, 0, cipher.size(), cipher_data);
}

void NgramsToDevice(const string & uni_filename,        
  const string & bi_filename, const string & tri_filename)
{
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
  oclCmdQueue.writeBuffer(d_plugsBuffer, 0, ALPSIZE, plugboard.plugs);
  oclCmdQueue.writeBuffer(d_fixedBuffer, 0, ALPSIZE, plugboard.fixed);
}

void SetUpResultsMemory(int count)
{
  resultsBuffer = clpp::Buffer(oclContext, CL_MEM_READ_WRITE, count*sizeof(Result));
  FindBestResultKernel.setArg(0, resultsBuffer);
}

void InitializeArrays(const string cipher_string, int turnover_modes,        
  int score_kinds, int digits)
{
  //d_ciphertext
  CipherTextToDevice(cipher_string);
  //d_wiring
  SetUpScramblerMemory();
  //allow_turnover
  h_task.turnover_modes = turnover_modes;
  //use unigrams
  h_task.score_kinds = score_kinds;

  //d_results
  int count = (int)pow(ALPSIZE, digits);
  SetUpResultsMemory(count);
}

Result Climb(int cipher_length, const Key & key, bool single_key)
{
  oclCmdQueue.writeBuffer(d_keyBuffer, 0, sizeof(Key), &key);
  int grid_size = single_key ? 1 : ALPSIZE_TO3;
  int block_size = std::max(int(ClimbKernelWGSize), cipher_length);
  int shared_scrambler_size = ((cipher_length + 32) & ~31) * 28;
  ClimbKernel.setArg(10, clpp::Local(shared_scrambler_size));
  oclCmdQueue.enqueueNDRangeKernel(ClimbKernel, grid_size*block_size, block_size).wait();
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
  block_size = (count < REDUCE_MAX_THREADS * 2) ? nextPow2((count + 1) / 2) : REDUCE_MAX_THREADS;
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
  }
  
  FindBestResultKernel.setArg(2, cl_uint(count));
  oclCmdQueue.enqueueNDRangeKernel(FindBestResultKernel,
                            grid_size*block_size, block_size).wait();
  
  int s = grid_size;
  while (s > 1)
  {
    oclCmdQueue.enqueueCopyBuffer(resultsBuffer, d_tempBuffer, 0, 0,
                      s * sizeof(Result)).wait();
    ComputeDimensions(s, grid_size, block_size);
    FindBestResultKernel.setArg(2, cl_uint(s));
    oclCmdQueue.enqueueNDRangeKernel(FindBestResultKernel, grid_size*block_size,
                          block_size).wait();
    s = (s + (block_size * 2 - 1)) / (block_size * 2);
  }
  Result result;
  oclCmdQueue.readBuffer(d_tempBuffer, 0, sizeof(Result), &result);
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

  for (int i = 0; i < result.length(); i++)
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
