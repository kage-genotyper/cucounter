#include <iostream>
#include <sstream>
#include <inttypes.h>
#include <string>

#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"
#include "hashtable.h"

HashTable::HashTable(const uint64_t *keys, const bool cuda_keys, 
    const uint32_t size, const uint32_t capacity) 
{
  initialize(keys, cuda_keys, size, capacity);
}

void HashTable::initialize(const uint64_t *keys, const bool cuda_keys, 
    const uint32_t size, const uint32_t capacity) 
{
  capacity_m = capacity;
  size_m = size;

  // Allocate the table
  cuda_errchk(cudaMalloc(&table_m.keys, sizeof(uint64_t)*capacity));
  cuda_errchk(cudaMemset(table_m.keys, 0xff, sizeof(uint64_t)*capacity));
  cuda_errchk(cudaMalloc(&table_m.values, sizeof(uint32_t)*capacity));
  cuda_errchk(cudaMemset(table_m.values, 0, sizeof(uint32_t)*capacity));

  uint64_t *d_keys;
  if (!cuda_keys) 
  {
    cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*size));
    cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*size, cudaMemcpyHostToDevice));
  }

  // Synchronize because cudaMemset is asynchronous with respect to host
  cuda_errchk(cudaDeviceSynchronize());

#ifdef _USE_COOPERATIVE_GROUPS
  kernels::cg_initialize_hashtable(table_m, cuda_keys ? keys : d_keys, size, capacity);
#else
  kernels::initialize_hashtable(table_m, cuda_keys ? keys : d_keys, size, capacity);
#endif

  if (!cuda_keys) 
  {
    cuda_errchk(cudaFree(d_keys));
  }
}

void HashTable::get(const uint64_t *keys, uint32_t *counts, uint32_t size) const 
{
  uint64_t *d_keys;
  uint32_t *d_counts;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*size));
  cuda_errchk(cudaMalloc(&d_counts, sizeof(uint32_t)*size));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*size, cudaMemcpyHostToDevice));

#ifdef _USE_COOPERATIVE_GROUPS
  kernels::cg_lookup_hashtable(table_m, d_keys, d_counts, size, capacity_m); 
#else
  kernels::lookup_hashtable(table_m, d_keys, d_counts, size, capacity_m); 
#endif

  cuda_errchk(cudaMemcpy(counts, d_counts, sizeof(uint32_t)*size, cudaMemcpyDeviceToHost));
  cuda_errchk(cudaFree(d_keys));
  cuda_errchk(cudaFree(d_counts));
}

void HashTable::cu_get(const uint64_t *keys, uint32_t *counts, uint32_t size) const 
{
#ifdef _USE_COOPERATIVE_GROUPS
  kernels::cg_lookup_hashtable(table_m, keys, counts, size, capacity_m); 
#else
  kernels::lookup_hashtable(table_m, keys, counts, size, capacity_m); 
#endif
}

void HashTable::count(const uint64_t *keys, const uint32_t size,
    const bool count_revcomps, const uint8_t kmer_size) 
{
  uint64_t *d_keys;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*size));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*size, cudaMemcpyHostToDevice));

#ifdef _USE_COOPERATIVE_GROUPS
  kernels::cg_count_hashtable(table_m, d_keys, size, capacity_m, count_revcomps, kmer_size);
#else
  kernels::count_hashtable(table_m, d_keys, size, capacity_m, count_revcomps, kmer_size);
#endif

  cuda_errchk(cudaFree(d_keys));
}

void HashTable::cu_count(const uint64_t *keys, const uint32_t size,
    const bool count_revcomps, const uint8_t kmer_size) 
{
#ifdef _USE_COOPERATIVE_GROUPS
  kernels::cg_count_hashtable(table_m, keys, size, capacity_m, count_revcomps, kmer_size);
#else
  kernels::count_hashtable(table_m, keys, size, capacity_m, count_revcomps, kmer_size);
#endif
}

void HashTable::get_probe_lengths(
    const uint64_t *keys, uint32_t *lengths, const uint32_t size) const 
{
  uint64_t *d_keys;
  uint32_t *d_lengths;
  cuda_errchk(cudaMalloc(&d_keys, sizeof(uint64_t)*size));
  cuda_errchk(cudaMalloc(&d_lengths, sizeof(uint32_t)*size));
  cuda_errchk(cudaMemcpy(d_keys, keys, sizeof(uint64_t)*size, cudaMemcpyHostToDevice));

  kernels::get_probe_lengths(table_m, d_keys, d_lengths, size, capacity_m);
  cuda_errchk(cudaMemcpy(lengths, d_lengths, sizeof(uint32_t)*size, cudaMemcpyDeviceToHost));

  cuda_errchk(cudaFree(d_keys));
  cuda_errchk(cudaFree(d_lengths));
}

void HashTable::cu_get_probe_lengths(
    const uint64_t *keys, uint32_t *lengths, const uint32_t size) const
{
  kernels::get_probe_lengths(table_m, keys, lengths, size, capacity_m);
}

std::string HashTable::to_string(const bool full) const 
{
  int print_size = (capacity_m < 40) ? capacity_m : 40;

  uint64_t *keys = new uint64_t[capacity_m];
  uint32_t *values = new uint32_t[capacity_m];
  cuda_errchk(cudaMemcpy(
        keys, table_m.keys, sizeof(uint64_t)*capacity_m, cudaMemcpyDeviceToHost));
  cuda_errchk(cudaMemcpy(
        values, table_m.values, sizeof(uint32_t)*capacity_m, cudaMemcpyDeviceToHost));

  std::ostringstream oss;
  std::ostringstream keys_oss;
  std::ostringstream values_oss;

  keys_oss << "[";
  values_oss << "[";
  uint32_t elements = 0;
  for (int i = 0; i < capacity_m; i++) 
  {
    uint64_t key = keys[i];
    uint32_t value = values[i];

    if (key == kEmpty && !full) { continue; }

    if (elements != 0) 
    { 
      keys_oss << ", "; 
      values_oss << ", "; 
    }

    if (key == kEmpty)
    {
      keys_oss << "kEmp";
    }
    else
    {
      keys_oss << key;
    }

    //keys_oss << ((key == kEmpty) ? "E" : key);
    values_oss << value;
    
    elements++;
    if (elements >= print_size && !full) { break; }
  }
  keys_oss << "]";
  values_oss << "]";

  oss << "Counter(" << keys_oss.str() << ", " << values_oss.str();
  oss << ", size=" << size_m << ", capacity=" << capacity_m << ")";

  delete[] keys;
  delete[] values;

  return oss.str();
}
