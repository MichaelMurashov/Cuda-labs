#include <iostream>
#include <time.h>
#include <omp.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

using namespace std;

const int block_size = 16;

__global__
void shared_kernel(int n, const float *a, const float *b, float *c) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * block_size * by;
    int aEnd = aBegin + n - 1;
    int bBegin = block_size * bx;
    int bStep = block_size * n;

    int aStep = block_size;

    float sum = 0.0f;

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        __shared__ float as[block_size][block_size];
        __shared__ float bs[block_size][block_size];

        as[ty][tx] = a[ia + n * ty + tx];
        bs[ty][tx] = b[ib + n * ty + tx];

        __syncthreads();

        for (int k = 0; k < block_size; k++)
            sum += as[ty][k] * bs[k][tx];

        __syncthreads();
    }

    int ic = n * block_size * by + block_size * bx;

    c[ic + n * ty + tx] = sum;
}

float sharedMultMat(int n, const float *dev_a, const float *dev_b, float *dev_c) {
    dim3 threads(block_size, block_size);
    dim3 blocks(n / threads.x, n / threads.y);

    clock_t start = clock();

    shared_kernel <<< blocks, threads >>> (n, dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    clock_t finish = clock();

    return (float)(finish - start) / CLOCKS_PER_SEC;
}

__global__
void kernel(int n, const float *a, const float *b, float *c) {
    int   bx = blockIdx.x;
    int   by = blockIdx.y;
    int   tx = threadIdx.x;
    int   ty = threadIdx.y;

    float sum = 0.0f;

    int   ia = n * block_size * by + n * ty;
    int   ib = block_size * bx + tx;

    for (int k = 0; k < n; k++)
        sum += a[ia + k] * b[ib + k*n];

    int ic = n * block_size * by + block_size * bx;
    c[ic + n * ty + tx] = sum;
}

float multMat(int n, const float *dev_a, const float *dev_b, float *dev_c) {
    //int num_block = (n + block_size - 1) / block_size;

    dim3 threads(block_size, block_size);
    dim3 blocks(n / threads.x, n / threads.y);

    clock_t start = clock();

    kernel <<< blocks, threads >>> (n, dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    clock_t finish = clock();

    return (float)(finish - start) / CLOCKS_PER_SEC;
}

float cpuMultMat(int n, const float *a, const float *b, float*c) {
    int i, j, k;
    int NestedThreadsNum = 2;

    omp_set_nested(true);
    omp_set_num_threads(NestedThreadsNum);

    clock_t start = clock();

    #pragma omp parallel for private (j, k)
    for (i = 0; i < n; i++)
        #pragma omp parallel for private (k)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
                c[i * n + j] += a[i * n + k] * b[k * n + j];

    clock_t finish = clock();

    return (float)(finish - start) / CLOCKS_PER_SEC;
}

int main() {
    const int n = 64;

    float *a = new float[n * n], *b = new float[n * n], *c = new float[n * n];
    float *resultGPU = new float[n * n], *resultSharedGPU = new float[n * n];

    float *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, n * n * sizeof(float));
    cudaMalloc((void**)&dev_b, n * n * sizeof(float));
    cudaMalloc((void**)&dev_c, n * n * sizeof(float));

    for (int i = 0; i < n * n; i++) {
        a[i] = b[i] = i;
        c[i] = 0;
    }

    cudaMemcpy(dev_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, n * n * sizeof(float), cudaMemcpyHostToDevice);

    float cpuTime = cpuMultMat(n, a, b, c);

    float gpuTime = multMat(n, dev_a, dev_b, dev_c);

    cudaMemcpy(resultGPU, dev_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    float sharedGpuTime = sharedMultMat(n, dev_a, dev_b, dev_c);

    cudaMemcpy(resultSharedGPU, dev_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    //for (int i = 0; i < n * n; i++)
    //    cout << c[i] << ' ';
    //cout << endl;

    //for (int i = 0; i < n * n; i++)
    //    cout << resultGPU[i] << ' ';
    //cout << endl;

    for (int i = 0; i < n * n; i++)
        if (c[i] != resultGPU[i]) {
            cout << "Matrixs cpu and gpu are not equal!" << endl;
            break;
        }

    for (int i = 0; i < n * n; i++)
        if (c[i] != resultSharedGPU[i]) {
            cout << "Matrixs cpu and shared_gpu are not equal!" << endl;
            break;
        }

    cout << "gpu time: " << gpuTime << endl
        << "shared gpu time: " << sharedGpuTime << endl
        << "cpu time: " << cpuTime << endl;

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    free(a); free(b); free(c);
    system("pause");

    return 0;
}
