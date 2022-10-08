#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define _DEBUG

static const uint64_t kEmpty = 0xffffffffffffffff;
static const uint32_t vInvalid = 0xffffffff;

#define cuda_errchk(err) { cuda_errcheck(err, __FILE__, __LINE__); }
inline void cuda_errcheck(cudaError_t code, const char *file, int line, bool abort=true) {
#ifdef _DEBUG
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA assert: '%s', in %s, at line %d\n", cudaGetErrorString(code), file, line);
    if (abort) { exit(code); }
  }
#endif // _DEBUG
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
