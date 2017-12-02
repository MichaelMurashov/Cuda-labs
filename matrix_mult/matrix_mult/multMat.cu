#include <time.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

using namespace std;

const int block_size = 256;

__global__
void multMat_kernel(int n, const float *a, const float *b, float *c) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0;

    int ia = n * block_size * by + n * ty;
    int ib = block_size * bx + tx;

    for (int k = 0; k < n; k++)
        sum += a[ia + k] * b[ib + k * n];

    int ic = n * block_size * by + block_size * bx;
    c[ic + n * ty + tx] = sum;
}

float multMat(int n, const float *dev_a, const float *dev_b, float *dev_c) {
    int num_block = (n + block_size - 1) / block_size;
    
    clock_t start = clock();
    
    multMat_kernel <<< num_block, block_size >>> (n, dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    clock_t finish = clock();

    return (float)(finish - start) / CLOCKS_PER_SEC;
}
