#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cmath>
#include <torch/extension.h>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n)+((m) % (n) > 0))

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
    CHECK_CUDA(x); \
    CHECK_CONTIGUOUS(x)

/***
 * calculate proper thread number
 * If work_size < TOTAL_THREADS, number = work_size (2^n)
 * Else number = TOTAL_THREADS
***/
inline int opt_n_threads(int work_size) {
    // log2(work_size)
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    // 1 * 2^(pow_2)
    return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}

#endif