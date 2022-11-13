#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "common.h"
#include "kernels.h"

namespace kernels {

namespace cg = cooperative_groups;

__device__ inline uint64_t word_reverse_complement(const uint64_t kmer, uint8_t kmer_size) 
{
  uint64_t res = ~kmer;
  res = ((res >> 2 & 0x3333333333333333) | (res & 0x3333333333333333) << 2);
  res = ((res >> 4 & 0x0F0F0F0F0F0F0F0F) | (res & 0x0F0F0F0F0F0F0F0F) << 4);
  res = ((res >> 8 & 0x00FF00FF00FF00FF) | (res & 0x00FF00FF00FF00FF) << 8);
  res = ((res >> 16 & 0x0000FFFF0000FFFF) | (res & 0x0000FFFF0000FFFF) << 16);
  res = ((res >> 32 & 0x00000000FFFFFFFF) | (res & 0x00000000FFFFFFFF) << 32);
  return (res >> (2 * (32 - kmer_size)));
}

__global__ void init_hashtable_kernel(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) 
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) 
    {
      unsigned long long int *old_ptr = reinterpret_cast<unsigned long long int *>(&table.keys[hash]);
      uint64_t old = atomicCAS(old_ptr, kEmpty, key);

      if (old == kEmpty || old == key) 
      {
        table.values[hash] = 0;
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void init_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) 
{
  //int min_grid_size;
  int thread_block_size = 1024;
  //cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
  //    &min_grid_size, &thread_block_size, 
  //    init_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  init_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, size, capacity);
  //cuda_errchk(cudaDeviceSynchronize());
}

__global__ void lookup_hashtable_kernel(Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity) 
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) 
  {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) 
    {
      uint64_t cur_key = table.keys[hash];
      if (cur_key == key || cur_key == kEmpty) 
      {
        counts[thread_id] = (cur_key == key) ? table.values[hash] : 0;
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void lookup_hashtable(Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity) 
{
  //int min_grid_size;
  int thread_block_size = 1024;
  //cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
  //    &min_grid_size, &thread_block_size, 
  //    init_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  lookup_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, counts, size, capacity);
  //cuda_errchk(cudaDeviceSynchronize());
}

__global__ void count_hashtable_kernel(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) 
  {
    // Search for original key
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true)
    {
      uint64_t cur_key = table.keys[hash];
      if (cur_key == kEmpty) 
      { 
        break; 
      }
      if (cur_key == key) 
      {
        atomicAdd((unsigned int *)&(table.values[hash]), 1);
        break;
      }
      hash = (hash + 1) % capacity;
    }

    if (count_revcomps)
    {
      // Search for reverse complement of key
      key = word_reverse_complement(key, kmer_size);
      hash = key % capacity;

      while (true) 
      {
        uint64_t cur_key = table.keys[hash];
        if (cur_key == kEmpty) 
        { 
          return;
        }
        if (cur_key == key) 
        {
          atomicAdd((unsigned int *)&(table.values[hash]), 1);
          return;
        }
        hash = (hash + 1) % capacity;
      }
    }
  }
}

void count_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  //int min_grid_size;
  int thread_block_size = 1024;
  //cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
  //    &min_grid_size, &thread_block_size, 
  //    init_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  count_hashtable_kernel<<<grid_size, thread_block_size>>>(
      table, keys, size, capacity, count_revcomps, kmer_size);
  //cuda_errchk(cudaDeviceSynchronize());
}

__global__ void cg_count_hashtable_kernel( 
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  int key_index = (blockIdx.x * blockDim.x + threadIdx.x) / CG_SIZE;

  if (key_index >= size) 
  {
    return;
  }

  cg::thread_block_tile<CG_SIZE> group = cg::tiled_partition<CG_SIZE>(cg::this_thread_block());
  uint64_t insert_key = keys[key_index];

  uint64_t hash = insert_key % capacity;
  hash = (hash + group.thread_rank()) % capacity;

  while (true) 
  {
    uint64_t table_key = table.keys[hash];

    const bool hit = (insert_key == table_key);
    const auto hit_mask = group.ballot(hit);

    if (hit_mask)
    {
      const int leader = __ffs(hit_mask) - 1;
      if (group.thread_rank() == leader)
      {
        atomicAdd((unsigned int *)&(table.values[hash]), 1);
      }
      return;
    }

    const bool empty = (table_key == kEmpty);
    const auto empty_mask = group.ballot(empty);

    if (empty_mask) {
      return;
    }

    hash = (hash + CG_SIZE) % capacity;
  }
}

void cg_count_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  //int min_grid_size;
  int thread_block_size = 1024;
  //cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
  //    &min_grid_size, &thread_block_size, 
  //    init_hashtable_kernel, 0, 0));

  int grid_size = (size*CG_SIZE) / thread_block_size + ((size*CG_SIZE) % thread_block_size > 0);
  count_hashtable_kernel<<<grid_size, thread_block_size>>>(
      table, keys, size, capacity, count_revcomps, kmer_size);
  //cuda_errchk(cudaDeviceSynchronize());
}

} // kernels
