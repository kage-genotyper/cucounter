#ifndef HASHTABLE_H_
#define HASHTABLE_H_

#include <sstream>
#include <inttypes.h>
#include <string>

#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"

class HashTable
{
public:
  HashTable() = default;
  HashTable(const uint64_t *keys, const bool cuda_keys, 
      const uint32_t size, const uint32_t capacity);
  ~HashTable() 
  { 
    cuda_errchk(cudaFree(table_m.keys)); 
    cuda_errchk(cudaFree(table_m.values)); 
  }

  uint32_t size() const { return size_m; }
  uint32_t capacity() const { return capacity_m; }

  void count(const uint64_t *keys, const uint32_t size, 
      const bool count_revcomps, const uint8_t kmer_size);
  void cu_count(const uint64_t *keys, const uint32_t size,
      const bool count_revcomps, const uint8_t kmer_size);

  void get(const uint64_t *keys, uint32_t *counts, uint32_t size) const;
  void cu_get(const uint64_t *keys, uint32_t *counts, uint32_t size) const;

  void get_probe_lengths(const uint64_t *keys, uint32_t *lengths, const uint32_t size) const;
  void cu_get_probe_lengths(const uint64_t *keys, uint32_t *lengths, const uint32_t size) const;

  std::string to_string(const bool full) const;
private:
  uint32_t size_m;
  uint32_t capacity_m;
  Table table_m;

  void initialize(const uint64_t *keys, const bool cuda_keys, 
      const uint32_t size, const uint32_t capacity);
};

#endif // HASHTABLE_H_
