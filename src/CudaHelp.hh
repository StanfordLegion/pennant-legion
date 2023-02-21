
#ifndef __CUDA_HELP_H__
#define __CUDA_HELP_H__

#define THREADS_PER_BLOCK   128
#define MIN_CTAS_PER_SM     4
#define MAX_REDUCTION_CTAS  1024

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "resilience.h"
#define Legion ResilientLegion
#ifndef __CUDA_HD__
#define __CUDA_HD__ __host__ __device__
#endif

#ifdef __CUDACC__
template<typename REDUCTION>
__device__ __forceinline__
void reduce_double(Legion::DeferredReduction<REDUCTION> result, double value)
{
  __shared__ double trampoline[THREADS_PER_BLOCK/32];
  // Reduce across the warp
  const int laneid = threadIdx.x & 0x1f;
  const int warpid = threadIdx.x >> 5;
  for (int i = 16; i>= 1; i/=2) {
    const int lo_part = __shfl_xor_sync(0xffffffff, __double2loint(value), i, 32);
    const int hi_part = __shfl_xor_sync(0xffffffff, __double2hiint(value), i, 32);
    const double shuffle_value = __hiloint2double(hi_part, lo_part);
    REDUCTION::template fold<true/*exclusive*/>(value, shuffle_value);
  }
  // Write warp values into shared memory
  if ((laneid == 0) && (warpid > 0))
    trampoline[warpid] = value;
  __syncthreads();
  // Output reduction
  if (threadIdx.x == 0)
  {
    for (int i = 1; i < (THREADS_PER_BLOCK/32); i++)
      REDUCTION::template fold<true/*exclusive*/>(value, trampoline[i]);
    result <<= value;
    // Make sure the result is visible externally
    __threadfence_system();
  }
}
#endif

#else
#define __CUDA_HD__
#endif

#endif // __CUDA_HELP_H__

