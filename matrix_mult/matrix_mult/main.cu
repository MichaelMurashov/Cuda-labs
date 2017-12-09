#include <iostream>
#include <time.h>
#include <omp.h>
#include <iomanip>
#include <cmath>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda.h"

using namespace std;

#define block_size 64

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

    //clock_t start = clock();
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    shared_kernel <<< blocks, threads >>> (n, dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    //clock_t finish = clock();

    //return (float)(finish - start) / CLOCKS_PER_SEC;

    cudaEventDestroy(start); cudaEventDestroy(stop);

    return gpuTime;
}

__global__
void kernel_t(int n, const float *a, const float *b, float *c) {
    
    printf("from kernel\n");
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
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(n / dimBlock.x, n / dimBlock.y);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    int num_block = (n + block_size - 1) / block_size;
    kernel_t <<< num_block, block_size >>> (n, dev_a, dev_b, dev_c);
    //cudaDeviceSynchronize();

    //kernel_t <<< dimGrid, dimBlock >>> (n, dev_a, dev_b, dev_c);
    //cudaDeviceSynchronize();

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventDestroy(start); cudaEventDestroy(stop);

    return gpuTime;
}

float cpuMultMat(int n, const float *a, const float *b, float*c) {
    int i, j, k;

    clock_t start = clock();

    #pragma omp parallel for private (j, k)
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            float elem = 0.0f;
            for (k = 0; k < n; k++)
                elem += a[i * n + k] * b[k * n + j];
            c[i * n + j] = elem;
        }

    clock_t finish = clock();

    return (float)(finish - start) / CLOCKS_PER_SEC;
}

int main() {
    const int n = 128;
    const float e = 1e-5;

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

   // cudaMemcpy(resultSharedGPU, dev_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    //for (int i = 0; i < n * n; i++)
    //    cout << c[i] << " ";
    //cout << endl;

    //for (int i = 0; i < n * n; i++)
    //    cout << resultGPU[i] << " ";
    //cout << endl;

    //for (int i = 0; i < n * n; i++)
    //    cout << resultSharedGPU[i] << " ";
    //cout << endl;


    for (int i = 0; i < n * n; i++)
        if (std::abs(c[i] - resultGPU[i]) > e) {
            cout << "Matrixs cpu and gpu are not equal!" << endl;
            break;
        }

    //for (int i = 0; i < n * n; i++) {
    //    if (fabsf(c[i] - resultSharedGPU[i]) > e) {
    //        cout << "Matrixs cpu and shared_gpu are not equal!" << endl;
    //        break;
    //    }
    //}

    //cout << fixed << "gpu time: " << setprecision(10) << gpuTime << endl
    //    << "shared gpu time: " << sharedGpuTime << endl
    //    << "cpu time: " << cpuTime << endl;

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    delete[]a; free(b); free(c);
    system("pause");

    return 0;
}
