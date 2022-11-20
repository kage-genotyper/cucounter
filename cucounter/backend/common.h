#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>

#include <cuda_runtime.h>

#ifdef _COOPERATIVE_GROUP_SIZE
static const unsigned int cg_size = _COOPERATIVE_GROUP_SIZE;
#else
static const unsigned int cg_size = 4;
#endif

static const uint64_t kEmpty = 0xffffffffffffffff;

#define cuda_errchk(err) { cuda_errcheck(err, __FILE__, __LINE__); }

inline void cuda_errcheck(cudaError_t code, const char *file, int line, bool abort=true) 
{
#ifdef _CUDA_ERROR_CHECK
  if (code != cudaSuccess) 
  {
    switch (code) 
    {
      case 2:
        fprintf(stderr, "CUDA out of memory error in %s at line %d\n", file, line);
        break;
      default:
        fprintf(stderr, "CUDA assert: '%s', in %s, at line %d\n", cudaGetErrorString(code), file, line);
    }
    exit(code);
  }
#endif // _CUDA_ERROR_CHECK
}

struct Table 
{
  uint64_t *keys;
  uint32_t *values;
};

#endif // COMMON_H_
