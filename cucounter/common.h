#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define _ERR_CHECK 

static const uint64_t kEmpty = 0xffffffffffffffff;
static const uint32_t vInvalid = 0xffffffff;

#define cuda_errchk(err) { cuda_errcheck(err, __FILE__, __LINE__); }
inline void cuda_errcheck(cudaError_t code, const char *file, int line, bool abort=true) {
#ifdef _ERR_CHECK
  if (code != cudaSuccess) {
    switch (code) {
      case 2:
        fprintf(stderr, "CUDA out of memory error in %s at line %d\n", file, line);
        exit(code);
        break;
      default:
        fprintf(stderr, "CUDA assert: '%s', in %s, at line %d\n", cudaGetErrorString(code), file, line);
    }
  }
#endif // _ERR_CHECK
}

// No longer using this
struct KeyValue {
  uint64_t key;
  uint64_t value;
};

struct Table {
  uint64_t *keys;
  uint32_t *values;
};

#endif // COMMON_H_
