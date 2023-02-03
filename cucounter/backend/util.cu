#include <inttypes.h>
#include "common.h"
#include "util.h"

size_t get_free_cuda_memory()
{
  int device;
  cudaGetDevice(&device);
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  return free;
}
