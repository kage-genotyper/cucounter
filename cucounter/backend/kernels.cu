#include <inttypes.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "common.h"
#include "kernels.h"

namespace kernels {

namespace cg = cooperative_groups;

__device__ __forceinline__ static uint64_t word_reverse_complement(
    const uint64_t kmer, uint8_t kmer_size) 
{
  uint64_t res = ~kmer;
  res = ((res >> 2 & 0x3333333333333333) | (res & 0x3333333333333333) << 2);
  res = ((res >> 4 & 0x0F0F0F0F0F0F0F0F) | (res & 0x0F0F0F0F0F0F0F0F) << 4);
  res = ((res >> 8 & 0x00FF00FF00FF00FF) | (res & 0x00FF00FF00FF00FF) << 8);
  res = ((res >> 16 & 0x0000FFFF0000FFFF) | (res & 0x0000FFFF0000FFFF) << 16);
  res = ((res >> 32 & 0x00000000FFFFFFFF) | (res & 0x00000000FFFFFFFF) << 32);
  return (res >> (2 * (32 - kmer_size)));
}

__device__ __forceinline__ static uint64_t murmur_hash(uint64_t kmer) 
{
#ifdef __USE_MURMUR_HASH__
  kmer ^= kmer >> 33;
  kmer *= 0xff51afd7ed558ccd;
  kmer ^= kmer >> 33;
  kmer *= 0xc4ceb9fe1a85ec53;
  kmer ^= kmer >> 33;
#endif
  return kmer;
}

// ----- INITIALIZE -----

__global__ void initialize_hashtable_kernel(Table table, 
    const uint64_t *keys, const uint32_t size, const uint32_t capacity) 
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= size) 
  {
    return;
  }

  uint64_t insert_key = keys[thread_id];
  uint64_t hash = murmur_hash(insert_key) % capacity;

  while (true) 
  {
    unsigned long long int *table_key_ptr = 
      reinterpret_cast<unsigned long long int *>(&table.keys[hash]);
    uint64_t old = atomicCAS(table_key_ptr, kEmpty, insert_key);

    const bool inserted = (old == kEmpty || old == insert_key);

    if (inserted)
    {
      return;
    }
    hash = (hash + 1) % capacity;
  }
}

void initialize_hashtable(Table table, 
    const uint64_t *keys, const uint32_t size, const uint32_t capacity) 
{
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      initialize_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  initialize_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, size, capacity);
}

__global__ void cg_initialize_hashtable_kernel(Table table, 
    const uint64_t *keys, const uint32_t size, const uint32_t capacity) 
{
  int key_index = (blockIdx.x * blockDim.x + threadIdx.x) / cg_size;
  if (key_index >= size) 
  {
    return;
  }

  cg::thread_block_tile<cg_size> group = cg::tiled_partition<cg_size>(cg::this_thread_block());
  uint64_t insert_key = keys[key_index];
  uint64_t hash = murmur_hash(insert_key) % capacity;
  hash = (hash + group.thread_rank()) % capacity;

  while (true) 
  {
    uint64_t table_key = table.keys[hash];

    bool empty = (table_key == kEmpty);
    auto empty_mask = group.ballot(empty);
    while (empty_mask)
    {
      bool inserted = false;

      // Determine leader
      const int leader = __ffs(empty_mask) - 1;
      if (group.thread_rank() == leader)
      {
        unsigned long long int *table_key_ptr = 
          reinterpret_cast<unsigned long long int *>(&table.keys[hash]);
        const uint64_t old = atomicCAS(table_key_ptr, kEmpty, insert_key);

        inserted = (old == kEmpty || old == insert_key);
      }

      if (group.any(inserted))
      {
        return;
      }

      empty_mask ^= (1UL << leader);
    }

    hash = (hash + cg_size) % capacity;
  }
}

void cg_initialize_hashtable(Table table, 
    const uint64_t *keys, const uint32_t size, const uint32_t capacity) 
{
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      cg_initialize_hashtable_kernel, 0, 0));

  int grid_size = (size*cg_size) / thread_block_size + ((size*cg_size) % thread_block_size > 0);
  cg_initialize_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, size, capacity);
}

// ----- LOOKUP -----

__global__ void lookup_hashtable_kernel(const Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity) 
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= size)
  {
    return;
  }

  uint64_t lookup_key = keys[thread_id];
  uint64_t hash = murmur_hash(lookup_key) % capacity;

  while (true) 
  {
    uint64_t table_key = table.keys[hash];
    if (table_key == lookup_key || table_key == kEmpty) 
    {
      counts[thread_id] = (table_key == lookup_key) ? table.values[hash] : 0;
      return;
    }
    hash = (hash + 1) % capacity;
  }
}

void lookup_hashtable(const Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity) 
{
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      lookup_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  lookup_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, counts, size, capacity);
}

__global__ void cg_lookup_hashtable_kernel(const Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity)
{
  int key_index = (blockIdx.x * blockDim.x + threadIdx.x) / cg_size;
  if (key_index >= size) 
  {
    return;
  }

  cg::thread_block_tile<cg_size> group = cg::tiled_partition<cg_size>(cg::this_thread_block());
  uint64_t lookup_key = keys[key_index];
  uint64_t hash = murmur_hash(lookup_key) % capacity;
  hash = (hash + group.thread_rank()) % capacity;

  while (true) 
  {
    uint64_t table_key = table.keys[hash];

    const bool hit = (lookup_key == table_key);
    const auto hit_mask = group.ballot(hit);
    if (hit_mask)
    {
      const int leader = __ffs(hit_mask) - 1;
      if (group.thread_rank() == leader)
      {
        counts[key_index] = table.values[hash];
      }
      return;
    }

    const bool empty = (table_key == kEmpty);
    const auto empty_mask = group.ballot(empty);
    if (empty_mask) {
      return;
    }

    hash = (hash + cg_size) % capacity;
  }
}

void cg_lookup_hashtable(const Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity)
{
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      cg_lookup_hashtable_kernel, 0, 0));

  int grid_size = (size*cg_size) / thread_block_size + ((size*cg_size) % thread_block_size > 0);
  lookup_hashtable_kernel<<<grid_size, thread_block_size>>>(
      table, keys, counts, size, capacity);
}

// ----- COUNT -----

__global__ void count_hashtable_kernel(Table table, 
    const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= size)
  {
    return;
  }

  // Search for original key
  uint64_t insert_key = keys[thread_id];
  uint64_t hash = murmur_hash(insert_key) % capacity;

  while (true)
  {
    uint64_t table_key = table.keys[hash];
    if (table_key == kEmpty) 
    { 
      break; 
    }
    if (table_key == insert_key) 
    {
      atomicAdd((unsigned int *)&(table.values[hash]), 1);
      break;
    }
    hash = (hash + 1) % capacity;
  }

  if (count_revcomps)
  {
    // Search for reverse complement of key
    insert_key = word_reverse_complement(insert_key, kmer_size);
    hash = murmur_hash(insert_key) % capacity;

    while (true) 
    {
      uint64_t table_key = table.keys[hash];
      if (table_key == kEmpty) 
      { 
        return;
      }
      if (table_key == insert_key) 
      {
        atomicAdd((unsigned int *)&(table.values[hash]), 1);
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void count_hashtable(Table table, 
    const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      count_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  count_hashtable_kernel<<<grid_size, thread_block_size>>>(
      table, keys, size, capacity, count_revcomps, kmer_size);
}

__global__ void cg_count_hashtable_kernel(Table table, 
    const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  int key_index = (blockIdx.x * blockDim.x + threadIdx.x) / cg_size;
  if (key_index >= size) 
  {
    return;
  }

  cg::thread_block_tile<cg_size> group = cg::tiled_partition<cg_size>(cg::this_thread_block());
  uint64_t insert_key = keys[key_index];
  uint64_t hash = murmur_hash(insert_key) % capacity;
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

    hash = (hash + cg_size) % capacity;
  }
}

void cg_count_hashtable(Table table, 
    const uint64_t *keys, const uint32_t size, const uint32_t capacity,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      cg_count_hashtable_kernel, 0, 0));

  int grid_size = (size*cg_size) / thread_block_size + ((size*cg_size) % thread_block_size > 0);
  count_hashtable_kernel<<<grid_size, thread_block_size>>>(
      table, keys, size, capacity, count_revcomps, kmer_size);
}

// ----- PROBE LENGHT -----

__global__ void get_probe_lengths_kernel(const Table table, 
    const uint64_t *keys, uint32_t *lengths, 
    const uint32_t size, const uint32_t capacity) 
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= size)
  {
    return;
  }

  uint64_t key = keys[thread_id];
  uint64_t hash = murmur_hash(key) % capacity;
  uint32_t probes = 1;

  while (true) 
  {
    uint64_t table_key = table.keys[hash];
    if (table_key == key || table_key == kEmpty) 
    {
      lengths[thread_id] = probes;
      return;
    }
    hash = (hash + 1) % capacity;
    probes++;
  }
}

void get_probe_lengths(const Table table, 
    const uint64_t *keys, uint32_t *lengths, 
    const uint32_t size, const uint32_t capacity)
{
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      get_probe_lengths_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  get_probe_lengths_kernel<<<grid_size, thread_block_size>>>(
      table, keys, lengths, size, capacity);
}

} // kernels
