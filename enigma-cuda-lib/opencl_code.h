#pragma once

/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <inttypes.h>
#include "wiring.h"
#include "plugboard.h"
  
enum ScoreKind {skIC=1, skUnigram=2, skBigram=4, skTrigram=8, skWords=16};

struct PitchedArray
{
    size_t pitch;
    int8_t * data;
};

struct Result
{
    int8_t plugs[ALPSIZE];
    int score;
    int index;
};

struct Task
{
    int count;
    PitchedArray scrambler;
    PitchedArray trigrams;
    int8_t turnover_modes;
    int score_kinds; 
    Result * results;
};

struct Block
{
    int count;
    int * trigrams;
    int8_t plugs[ALPSIZE];
    int unigrams[ALPSIZE];
    ScoreKind score_kind;
    int volatile score_buf[MAX_MESSAGE_LENGTH];
    int8_t plain_text[MAX_MESSAGE_LENGTH];
    int score;
};

extern void SetUpScramblerMemory();
extern void GenerateScrambler(const Key & key);
extern void CopyScramblerToHost(PitchedArray & dst);

extern void setUpConfig(int turnover_modes, int score_kinds, int cipher_length);
extern bool SelectGpuDevice(int req_major, int req_minor, int device, bool silent);
extern void CipherTextToDevice(string ciphertext_string);
extern void NgramsToDevice(const string & uni_filename,        
  const string & bi_filename, const string & tri_filename);
extern void OrderToDevice(const int8_t * order);
extern void PlugboardStringToDevice(string plugboard_string);
extern void PlugboardToDevice(const Plugboard & plugboard);
extern void SetUpResultsMemory(int count);
extern void InitializeArrays(const string cipher_string, int turnover_modes,
  int score_kinds, int digits = 3);
extern Result Climb(int cipher_length, const Key & key, bool single_key);
extern Result GetBestResult(int count);
extern string DecodeMessage(const string & ciphertext,
  const string & key_string, const string & plugboard_string);
extern string DecodeMessage(const string & ciphertext,
  const string & key_string, const int8_t * plugs);
extern int8_t DecodeLetter(int8_t c, const Key & key, const int8_t * plugs);

extern "C" {
int8_t mod26(const int16_t x);

extern unsigned char ___enigma_cuda_lib_opencl_program_cl[];
extern unsigned int ___enigma_cuda_lib_opencl_program_cl_len;
extern unsigned char ___enigma_cuda_lib_climb_clrx[];
extern unsigned int ___enigma_cuda_lib_climb_clrx_len;

};
