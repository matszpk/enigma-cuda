/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */
/* Copyright (c) 2017 Mateusz Szpakowski                               */

#define ALPSIZE 26
#define HALF_ALPSIZE (ALPSIZE / 2)
#define ALPSIZE_TO2 (26 * 26)
#define ALPSIZE_TO3 (26 * 26 * 26)
#define ALPSIZE_TO4 (26 * 26 * 26 * 26)
#define ALPSIZE_TO5 (26 * 26 * 26 * 26 * 26)

#define MIN_MESSAGE_LENGTH 18
#define MAX_MESSAGE_LENGTH 256

#define REDUCE_MAX_THREADS 256

#define NGRAM_DATA_TYPE int
#ifdef SHORT_BIGRAMS
#define NGRAM_DATA_TYPE_BIGRAM ushort
#else
#define NGRAM_DATA_TYPE_BIGRAM int
#endif
#ifdef SHORT_TRIGRAMS
#define NGRAM_DATA_TYPE_TRIGRAM ushort
#else
#define NGRAM_DATA_TYPE_TRIGRAM int
#endif

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed long int64_t;
typedef unsigned long uint64_t;

typedef enum _EnigmaModel { enigmaInvalid, enigmaHeeres, enigmaM3, enigmaM4, ENIGMA_MODEL_CNT } EnigmaModel;
typedef enum _RotorType { rotNone, rotI, rotII, rotIII, rotIV, rotV, rotVI, rotVII, rotVIII, rotBeta, rotGamma, ROTOR_TYPE_CNT } RotorType;
typedef enum _ReflectorType { refA, refB, refC, refB_thin, refC_thin, REFLECTOR_TYPE_CNT } ReflectorType;

typedef enum _ScoreKind {skIC=1, skUnigram=2, skBigram=4, skTrigram=8, skWords=16} ScoreKind;

#define NONE -1

typedef struct _Wiring {
	int8_t reflectors[REFLECTOR_TYPE_CNT][ALPSIZE];
	int8_t rotors[ROTOR_TYPE_CNT][ALPSIZE];
	int8_t reverse_rotors[ROTOR_TYPE_CNT][ALPSIZE];
	int8_t notch_positions[ROTOR_TYPE_CNT][2];
} Wiring;

typedef struct _ScramblerStructure
{
    EnigmaModel model;
    ReflectorType ukwnum;

    RotorType g_slot;
    RotorType l_slot;
    RotorType m_slot;
    RotorType r_slot;
} ScramblerStructure;

typedef struct _RotorSettings
{
    int g_ring;
    int l_ring;
    int m_ring;
    int r_ring;

    int g_mesg;
    int l_mesg;
    int m_mesg;
    int r_mesg;
} RotorSettings;

typedef enum _TurnoverLocation { toBeforeMessage = 1, toDuringMessage = 2, toAfterMessage = 4 } TurnoverLocation;

typedef struct _Key 
{
    ScramblerStructure stru;
    RotorSettings sett;
} Key;

typedef struct _Result
{
    union {
      uint32_t plugsW[(ALPSIZE+3)/4];
      int8_t plugs[ALPSIZE];
    };
    int score;
    int index;
} Result;

typedef struct _ResultScore
{
    int score;
    int index;
} ResultScore;

typedef struct _Block
{
    int count;
    int8_t plugs[ALPSIZE];
    int unigrams[ALPSIZE];
    ScoreKind score_kind;
    int volatile score_buf[MAX_MESSAGE_LENGTH];
    int8_t plain_text[MAX_MESSAGE_LENGTH];
    int score;
} Block;

inline int8_t mod26(const int16_t x)
{
  return (ALPSIZE * 2 + x) % ALPSIZE;
}

kernel void GenerateScramblerKernel(const constant Wiring* d_wiring,
            const constant Key* d_key, const uint thblockShift, const uint localShift,
            const uint scramblerDataPitch, global int8_t* scramblerData)
{
  const uint lid = get_local_id(0);
  const uint gidx = get_group_id(0);
  const uint gidy = get_group_id(1);
  const uint gidz = get_group_id(2);
  const uint thBlockNum = (1U<<thblockShift);
  const uint thblockMask = (1U<<(localShift-thblockShift)) - 1U;
  const constant int8_t * reflector;
  
  const constant int8_t * g_rotor;
  const constant int8_t * l_rotor;
  const constant int8_t * m_rotor;
  const constant int8_t * r_rotor;
  const constant int8_t * g_rev_rotor;
  const constant int8_t * l_rev_rotor;
  const constant int8_t * m_rev_rotor;
  const constant int8_t * r_rev_rotor;
  
  int8_t r_core_position;
  int8_t m_core_position;
  int8_t l_core_position;
  int8_t g_core_position;
  
  global int8_t * entry;
  
  uint thid = lid & thblockMask;
  if (thid < ALPSIZE)
  {
    reflector = d_wiring->reflectors[d_key->stru.ukwnum];
    
    g_rotor = d_wiring->rotors[d_key->stru.g_slot];
    l_rotor = d_wiring->rotors[d_key->stru.l_slot];
    m_rotor = d_wiring->rotors[d_key->stru.m_slot];
    r_rotor = d_wiring->rotors[d_key->stru.r_slot];
    
    g_rev_rotor = d_wiring->reverse_rotors[d_key->stru.g_slot];
    l_rev_rotor = d_wiring->reverse_rotors[d_key->stru.l_slot];
    m_rev_rotor = d_wiring->reverse_rotors[d_key->stru.m_slot];
    r_rev_rotor = d_wiring->reverse_rotors[d_key->stru.r_slot];
    
    //core positions
    r_core_position = gidx*thBlockNum + (lid>>(localShift-thblockShift));
    m_core_position = gidy;
    l_core_position = gidz;
    g_core_position = mod26(d_key->sett.g_mesg - d_key->sett.g_ring);
    
    entry = scramblerData + scramblerDataPitch  * (
      l_core_position *  ALPSIZE * ALPSIZE +
      m_core_position *  ALPSIZE +
      r_core_position);
    
    // after sync
    int8_t ch_in = thid;
    int8_t ch_out = ch_in;
    
    ch_out = r_rotor[mod26(ch_out + r_core_position)] - r_core_position;
    ch_out = m_rotor[mod26(ch_out + m_core_position)] - m_core_position;
    ch_out = l_rotor[mod26(ch_out + l_core_position)] - l_core_position;
    
    if (d_key->stru.model == enigmaM4)
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
}

#ifndef WITHOUT_CLIMB_KERNEL

int ComputeScramblerIndex(int char_pos, 
  const constant ScramblerStructure* stru,
  const local RotorSettings* sett, const constant Wiring* wiring)
{
  //retrieve notch info
  const constant int8_t * r_notch = wiring->notch_positions[stru->r_slot];
  const constant int8_t * m_notch = wiring->notch_positions[stru->m_slot];

  //period of the rotor turnovers
  int m_period = (r_notch[1] == NONE) ? ALPSIZE : HALF_ALPSIZE;
  int l_period = (m_notch[1] == NONE) ? ALPSIZE : HALF_ALPSIZE;
  l_period = (l_period-1) * m_period;

  //current wheel position relative to the last notch
  int r_after_notch = sett->r_mesg - r_notch[0];
  if (r_after_notch < 0) r_after_notch += ALPSIZE;
  if (r_notch[1] != NONE && r_after_notch >= (r_notch[1] - r_notch[0]))
    r_after_notch -= r_notch[1] - r_notch[0];

  int m_after_notch = sett->m_mesg - m_notch[0];
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
  return mod26(sett->l_mesg - sett->l_ring + l_steps) * ALPSIZE_TO2 +
    mod26(sett->m_mesg - sett->m_ring + m_steps) * ALPSIZE +
    mod26(sett->r_mesg - sett->r_ring + char_pos + 1);
}

TurnoverLocation GetTurnoverLocation(const constant ScramblerStructure* stru,
  const local RotorSettings* sett, int ciphertext_length, const constant Wiring* wiring)
{
  //rotors with two notches
    if (stru->r_slot > rotV && sett->r_ring >= HALF_ALPSIZE) 
        return toAfterMessage;
    if (stru->m_slot > rotV && sett->m_ring >= HALF_ALPSIZE) 
        return toAfterMessage;

  //does the left hand rotor turn right before the message?
  int8_t l_core_before = mod26(sett->l_mesg - sett->l_ring);
    int8_t l_core_first = ComputeScramblerIndex(0, stru, sett, wiring)
        / ALPSIZE_TO2;
  if (l_core_first != l_core_before) return toBeforeMessage;

  //does it turn during the message?
    int8_t l_core_last = 
        ComputeScramblerIndex(ciphertext_length-1, stru, sett, wiring) 
        / ALPSIZE_TO2;
  if (l_core_last != l_core_first) return toDuringMessage;

  return toAfterMessage;
}

local int8_t * ScramblerToShared(const global int8_t * global_scrambling_table,
        local int8_t* shared_scrambling_table, uint lid)
{
  //global: ALPSIZE bytes at sequential addresses
  const global int32_t * src = (const global int32_t *)(global_scrambling_table);

  //shared: same bytes in groups of 4 at a stride of 128
  //extern __shared__ int8_t shared_scrambling_table[];
  local int32_t * dst = (local int32_t *)(shared_scrambling_table);

  //copy ALPSIZE bytes as 7 x 32-bit words
  int idx = (lid & ~(SCRAMBLER_STRIDE-1)) * 7 + (lid & (SCRAMBLER_STRIDE-1));
  for (int i = 0; i < 7; ++i) dst[idx + SCRAMBLER_STRIDE * i] = src[i];
  return &shared_scrambling_table[idx * 4];
}

int8_t Decode(const local int8_t * plugboard, const local int8_t * scrambling_table,
        const constant int8_t* d_ciphertext, uint lid)
{
  int8_t c = d_ciphertext[lid];
  c = plugboard[c];
  c = scrambling_table[(c & ~3) * SCRAMBLER_STRIDE + (c & 3)];
  c = plugboard[c];  
  return c;
}

// especially optimized for specific wavefront size
void Sum(int count, volatile local int * data, local int * sum, uint lid)
{
#if WAVEFRONT_SIZE>=128
  {
    if ((lid + 128) < count) data[lid] += data[128 + lid];
    if ((lid + 64) < count) data[lid] += data[64 + lid];
    if ((lid + 32) < count) data[lid] += data[32 + lid];
    if ((lid + 16) < count) data[lid] += data[16 + lid];
#elif WAVEFRONT_SIZE>=64
  if ((lid + 128) < count) data[lid] += data[128 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 64)
  {
    if ((lid + 64) < count) data[lid] += data[64 + lid];
    if ((lid + 32) < count) data[lid] += data[32 + lid];
    if ((lid + 16) < count) data[lid] += data[16 + lid];
#elif WAVEFRONT_SIZE>=32
  if ((lid + 128) < count) data[lid] += data[128 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 64 && (lid + 64) < count) data[lid] += data[64 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 32)
  {
    if ((lid + 32) < count) data[lid] += data[32 + lid];
    if ((lid + 16) < count) data[lid] += data[16 + lid];
#elif WAVEFRONT_SIZE>=16
  if ((lid + 128) < count) data[lid] += data[128 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 64 && (lid + 64) < count) data[lid] += data[64 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 32 && (lid + 32) < count) data[lid] += data[32 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 16)
  {
    if ((lid + 16) < count) data[lid] += data[16 + lid];
#elif WAVEFRONT_SIZE>=8
  if ((lid + 128) < count) data[lid] += data[128 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 64 && (lid + 64) < count) data[lid] += data[64 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 32 && (lid + 32) < count) data[lid] += data[32 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 16 && (lid + 16) < count) data[lid] += data[16 + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 8)
  {
#else
    #error "Wavefront smaller than 8 workitems is not supported!"
#endif
    data[lid] += data[8 + lid];
    data[lid] += data[4 + lid];
    data[lid] += data[2 + lid];
    if (lid == 0) *sum = data[0] + data[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

#define HISTO_SIZE 32

void IcScore(local Block * block, const local int8_t * scrambling_table,
             const constant int8_t* d_ciphertext, uint lid)
{
  //init histogram
  if (lid < HISTO_SIZE) block->score_buf[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  //compute histogram
  if (lid < block->count)
  {
    int8_t c = Decode(block->plugs, scrambling_table, d_ciphertext, lid);
    //atomicAdd((int *)&block->score_buf[c], 1);
    atomic_add(&block->score_buf[c], 1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  //TODO: try lookup table here, ic[MAX_MESSAGE_LENGTH]
  if (lid < HISTO_SIZE)
    block->score_buf[lid] *= block->score_buf[lid] - 1;

  //sum up
#if WAVEFRONT_SIZE>=16
#  if WAVEFRONT_SIZE<32
  barrier(CLK_LOCAL_MEM_FENCE);
#  endif
  if (lid < (HISTO_SIZE >> 1))
  {
    block->score_buf[lid] += block->score_buf[lid + 16];
#elif WAVEFRONT_SIZE>=8
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < (HISTO_SIZE >> 1)) block->score_buf[lid] += block->score_buf[lid + 16];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 8)
  {
#else
    #error "Wavefront smaller than 8 workitems is not supported!"
#endif
    block->score_buf[lid] += block->score_buf[lid + 8];
    block->score_buf[lid] += block->score_buf[lid + 4];
    block->score_buf[lid] += block->score_buf[lid + 2];
    if (lid == 0) block->score = block->score_buf[0] + block->score_buf[1];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
}

void UniScore(local Block * block, const local int8_t * scrambling_table,
                const constant int8_t* d_ciphertext, uint lid)
{
  if (lid < block->count)
  {
    int8_t c = Decode(block->plugs, scrambling_table, d_ciphertext, lid);
    block->score_buf[lid] = block->unigrams[c];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  Sum(block->count, block->score_buf, &block->score, lid);
}

void BiScore(local Block * block, const local int8_t* scrambling_table,
              const constant int8_t* d_ciphertext,
              const constant NGRAM_DATA_TYPE_BIGRAM* d_bigrams, uint lid)
{
  if (lid < block->count)
    block->plain_text[lid] = Decode(block->plugs, scrambling_table, d_ciphertext, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  //TODO: trigrams are faster than bigrams. 
  //is it because trigrams are not declared as constants?
  //or because their index is computed explicitly?
  if (lid < (block->count - 1))
    block->score_buf[lid] = 
      d_bigrams[block->plain_text[lid]*ALPSIZE +
               block->plain_text[lid + 1]];
  barrier(CLK_LOCAL_MEM_FENCE);

  Sum(block->count - 1, block->score_buf, &block->score, lid);
}

void TriScore(local Block * block, const local int8_t* scrambling_table,
                  const constant int8_t* d_ciphertext,
                  const global NGRAM_DATA_TYPE_TRIGRAM* trigrams, uint lid)
{
  //decode char
  if (lid < block->count) 
    block->plain_text[lid] = Decode(block->plugs, scrambling_table, d_ciphertext, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  //look up scores
  if (lid < (block->count - 2))
    block->score_buf[lid] = trigrams[
      block->plain_text[lid] * ALPSIZE_TO2 +
      block->plain_text[lid + 1] * ALPSIZE +
      block->plain_text[lid+2]];
  barrier(CLK_LOCAL_MEM_FENCE);
  
  Sum(block->count - 2, block->score_buf, &block->score, lid);
}

void CalculateScore(local Block * block, const local int8_t * scrambling_table,
          const constant int8_t* d_ciphertext,
          const constant NGRAM_DATA_TYPE_BIGRAM* d_bigrams,
          const global NGRAM_DATA_TYPE_TRIGRAM* trigrams, uint lid)
{
  switch (block->score_kind)
  {
  case skTrigram: TriScore(block, scrambling_table, d_ciphertext, trigrams, lid); break;
  case skBigram:  BiScore(block, scrambling_table, d_ciphertext, d_bigrams, lid); break;
  case skUnigram: UniScore(block, scrambling_table, d_ciphertext, lid); break;
  case skIC:      IcScore(block, scrambling_table, d_ciphertext, lid); break;
  }
}

//------------------------------------------------------------------------------
//                               climber
//------------------------------------------------------------------------------
void TrySwap(int8_t i, int8_t k, const local int8_t * scrambling_table,
          local Block * block, const constant int8_t* d_ciphertext,
          const constant NGRAM_DATA_TYPE_BIGRAM* d_bigrams,
          const global NGRAM_DATA_TYPE_TRIGRAM* trigrams,
          local int* old_score, const constant int8_t* d_fixed, uint lid)
{
  int8_t x, z;
  *old_score = block->score;

  if (d_fixed[i] || d_fixed[k]) return;

  if (lid == 0)
  {              
    x = block->plugs[i];
    z = block->plugs[k];
    if (x != k)
    {
      block->plugs[x] = (x != i) ? x : block->plugs[x];
      block->plugs[z] = (z != k) ? z : block->plugs[z];
    }
    block->plugs[i] = (x==k) ? i : k;
    block->plugs[k] = (x==k) ? k : i;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  CalculateScore(block, scrambling_table, d_ciphertext, d_bigrams, trigrams, lid);

  if (lid == 0 && block->score <= *old_score)
  {
    block->score = *old_score;

    block->plugs[z] = k;
    block->plugs[x] = i;
    block->plugs[k] = z;
    block->plugs[i] = x;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

void MaximizeScore(local Block * block, const local int8_t * scrambling_table,
          const constant int8_t* d_ciphertext,
          const constant NGRAM_DATA_TYPE_BIGRAM* d_bigrams,
          const global NGRAM_DATA_TYPE_TRIGRAM* trigrams,
          local int* old_score, const constant int8_t* d_order,
          const constant int8_t* d_fixed, uint lid)
{
  CalculateScore(block, scrambling_table, d_ciphertext, d_bigrams, trigrams, lid);

  for (int p = 0; p < ALPSIZE - 1; p++)
    for (int q = p + 1; q < ALPSIZE; q++)
      TrySwap(d_order[p], d_order[q], scrambling_table, block, d_ciphertext, d_bigrams,
              trigrams, old_score, d_fixed, lid);
}


kernel void ClimbKernel(const constant Wiring* d_wiring,
            const constant Key* d_key, int taskCount,
            const uint scramblerDataPitch, const global int8_t* scramblerData,
            const uint trigramsDataPitch, const global NGRAM_DATA_TYPE_TRIGRAM* trigramsData,
            const constant NGRAM_DATA_TYPE* d_unigrams,
            const constant NGRAM_DATA_TYPE_BIGRAM* d_bigrams,
            const constant int8_t* d_plugs, const constant int8_t* d_order,
            const constant int8_t* d_fixed, const constant int8_t* d_ciphertext,
            global Result* taskResults,
            int turnover_modes, int score_kinds, local int8_t* shared_scrambling_table)
{
  local Block block;
  local RotorSettings sett;
  local bool skip_this_key;
  global Result * result;
  const uint lid = get_local_id(0);
  const uint gidx = get_group_id(0);
  const uint gidy = get_group_id(1);
  const uint gidz = get_group_id(2);
  const uint gxnum = get_num_groups(0);
  const uint gynum = get_num_groups(1);
  int linear_idx;
  
  if (lid < ALPSIZE)
  {
    block.plugs[lid] = d_plugs[lid];
    block.unigrams[lid] = d_unigrams[lid];
  }
  
  if (lid == 0)
  {
    block.count = taskCount;
    
    //ring and rotor settings to be tried
    sett.g_ring = 0;
    sett.l_ring = 0;

    //depending on the grid size, ring positions 
    //either from grid index or fixed (from d_key)
    sett.m_ring = (gynum > ALPSIZE) ? gidy / ALPSIZE : d_key->sett.m_ring;
    sett.r_ring = (gynum > 1) ? gidy % ALPSIZE : d_key->sett.r_ring;

    sett.g_mesg = d_key->sett.g_mesg;
    sett.l_mesg = (gxnum > ALPSIZE_TO2) ? gidx / ALPSIZE_TO2 : d_key->sett.l_mesg;
    sett.m_mesg = (gxnum > ALPSIZE) ? (gidx / ALPSIZE) % ALPSIZE : d_key->sett.m_mesg;
    sett.r_mesg = (gxnum > 1) ? gidx % ALPSIZE : d_key->sett.r_mesg;
  }
  {
    //element of results[] to store the output 
    linear_idx = gidz * ALPSIZE_TO2 + gidy * ALPSIZE + gidx;
    result = &taskResults[linear_idx];
    result->index = linear_idx;
    result->score = -1;
  }
  if (lid == 0)
  {
    skip_this_key = ((gxnum > 1) &&
      (GetTurnoverLocation(&(d_key->stru), &sett, block.count, d_wiring)
        & turnover_modes) == 0);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if (skip_this_key) return;
  
  const global int8_t * g_scrambling_table;
  local int8_t * scrambling_table;
  if (lid < block.count)
    {
      g_scrambling_table = scramblerData + 
      ComputeScramblerIndex(lid, &(d_key->stru), &sett, d_wiring) * 
        scramblerDataPitch;
      scrambling_table = ScramblerToShared(g_scrambling_table,
                        shared_scrambling_table, lid);
    }
  local int old_score;
  //IC once
  if (score_kinds & skIC)
  {
    block.score_kind = skIC;
    MaximizeScore(&block, scrambling_table, d_ciphertext, d_bigrams, trigramsData, &old_score,
              d_order, d_fixed, lid);
  }
  
  //unigrams once
  if (score_kinds & skUnigram)
  {
    block.score_kind = skUnigram;
    MaximizeScore(&block, scrambling_table, d_ciphertext, d_bigrams, trigramsData, &old_score,
              d_order, d_fixed, lid);
  }
  
  //bigrams once
  if (score_kinds & skBigram)
  {
    block.score_kind = skBigram;
    MaximizeScore(&block, scrambling_table, d_ciphertext, d_bigrams, trigramsData, &old_score,
              d_order, d_fixed, lid);
  }
  
  //trigrams until convergence
  if (score_kinds & skTrigram)
  {
    block.score_kind = skTrigram;
    block.score = 0;
    int pold_score;
    do
    {
      pold_score = block.score;
      MaximizeScore(&block, scrambling_table, d_ciphertext, d_bigrams, trigramsData,
                    &old_score, d_order, d_fixed, lid);
    } 
    while (block.score > pold_score);
  }
  
  //copy plugboard solution to global results array;
  if (lid < ALPSIZE) result->plugs[lid] = block.plugs[lid];
  if (lid == 0) result->score = block.score;
}

#endif

inline void SelectHigherScore(private Result* a, const global Result* b, uint bindex)
{
  if (b->score > a->score)
  {
    a->index = bindex;
    a->score = b->score;
  }
}

inline void SelectHigherScoreLocal(private Result* a, const local ResultScore* b)
{
  if (b->score > a->score)
  {
    a->index = b->index;
    a->score = b->score;
  }
}


kernel void FindBestResultKernel(const global Result* g_idata,
            global Result* g_odata, uint count)
{
  local ResultScore sdata[REDUCE_MAX_THREADS];
  unsigned int tid = get_local_id(0);
  unsigned int gid = get_group_id(0);
  unsigned int lsize = get_local_size(0);
  unsigned int i = gid*(lsize<<1) + tid;
  Result best_pair;
  if (i < count)
  {
    best_pair.index = i;
    best_pair.score = g_idata[i].score;
  }
  else
  {
    best_pair.index = count - 1;
    best_pair.score = g_idata[count-1].score;
  }
  
  if (i + lsize < count) SelectHigherScore(&best_pair, g_idata + i + lsize, i+lsize);
  
  sdata[tid].score = best_pair.score;
  sdata[tid].index = best_pair.index;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (unsigned int s = lsize >> 1; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      SelectHigherScoreLocal(&best_pair, sdata + tid + s);
      sdata[tid].score = best_pair.score;
      sdata[tid].index = best_pair.index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if (tid == 0) g_odata[gid] = g_idata[best_pair.index];
}
