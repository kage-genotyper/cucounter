#ifndef KERNELS_H_
#define KERNELS_H_

#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "common.h"

namespace kernels {

void init_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity);

void count_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size);

void cg_count_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size);

void lookup_hashtable(Table table, const uint64_t *keys, uint32_t *counts, 
    const uint32_t size, const uint32_t capacity); 

} // kernels

#endif // KERNELS_H_
