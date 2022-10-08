#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"

namespace kernels {

__global__ void init_hashtable_kernel(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) {
      unsigned long long int *old_ptr = reinterpret_cast<unsigned long long int *>(&table.keys[hash]);
      uint64_t old = atomicCAS(old_ptr, kEmpty, key);

      if (old == kEmpty || old == key) {
        table.values[hash] = 0;
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void init_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      init_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  init_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, size, capacity);
  cuda_errchk(cudaDeviceSynchronize());
}

__global__ void lookup_hashtable_kernel(Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) {
      uint64_t cur_key = table.keys[hash];
      if (cur_key == key || cur_key == kEmpty) {
        counts[thread_id] = (cur_key == key) ? table.values[hash] : 0;
        return;
      }
      hash = (hash + 1) % capacity;
    }
  }
}

void lookup_hashtable(Table table, 
    const uint64_t *keys, uint32_t *counts, const uint32_t size, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      lookup_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  lookup_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, counts, size, capacity);
  cuda_errchk(cudaDeviceSynchronize());
}

__global__ void count_hashtable_kernel(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id < size) {
    uint64_t key = keys[thread_id];
    uint64_t hash = key % capacity;

    while (true) {
      uint64_t cur_key = table.keys[hash];
      if (cur_key == kEmpty) { return; }
      if (cur_key == key) {
        atomicAdd((unsigned int *)&(table.values[hash]), 1);
        return;
      }

      hash = (hash + 1) % capacity;
    }
  }
}

void count_hashtable(
    Table table, const uint64_t *keys, const uint32_t size, const uint32_t capacity) {
  int min_grid_size;
  int thread_block_size;
  cuda_errchk(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &thread_block_size, 
      count_hashtable_kernel, 0, 0));

  int grid_size = size / thread_block_size + (size % thread_block_size > 0);
  count_hashtable_kernel<<<grid_size, thread_block_size>>>(table, keys, size, capacity);
  cuda_errchk(cudaDeviceSynchronize());
}

} // kernels
