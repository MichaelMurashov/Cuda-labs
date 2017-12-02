#include <iostream>
#include <cstdlib>
#include <time.h>
#include <omp.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

using namespace std;

__global__
void saxpy_kernel(int size, float alpha, float* x, int incx, float* y, int incy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        if (i * incx > size || i * incy > size)
            return;
        else
            y[i * incy] += alpha * x[i * incx];
}

template <typename T>
void axpy_cpu(int size, T alpha, const T *x, int incx, T *y, int incy) {
    for (int i = 0; i < size; i++)
        y[i * incy] += alpha * x[i * incx];
}

template <typename T>
void axpy_cpu_omp(int size, T alpha, const T *x, int incx, T *y, int incy) {
#pragma omp parallel for
    for (int i = 0; i < size; i++)
        y[i * incy] += alpha * x[i * incx];
}

int main() {
    int size = 200000000;
    float alpha = 1.0f;
    int incx = 1, incy = 1;

    float *x = new float[size], *x_gpu;
    float *y = new float[size], *y_gpu;
    float *result_gpu = new float[size];

    cudaMalloc((void**)&x_gpu, size * sizeof(float));
    cudaMalloc((void**)&y_gpu, size * sizeof(float));

    for (int i = 0; i < size; i++)
        x[i] = y[i] = i;

    cudaMemcpy(x_gpu, x, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y, size * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start_cpu = clock();
    axpy_cpu<float>(size, alpha, x, incx, y, incy);
    clock_t finish_cpu = clock();

    float time_cpu = (float)(finish_cpu - start_cpu) / CLOCKS_PER_SEC;

    clock_t start_cpu_omp = clock();
    axpy_cpu_omp<float>(size, alpha, x, incx, y, incy);
    clock_t finish_cpu_omp = clock();

    float time_cpu_omp = (float)(finish_cpu_omp - start_cpu_omp) / CLOCKS_PER_SEC;

    const int block_size = 256;
    int num_block = (size + block_size - 1) / block_size;

    clock_t start_gpu = clock();
    saxpy_kernel <<< num_block, block_size >>> (size, alpha, x_gpu, incx, y_gpu, incy);
    cudaDeviceSynchronize();
    clock_t finish_gpu = clock();

    float time_gpu = (float)(finish_gpu - start_gpu) / CLOCKS_PER_SEC;

    cudaMemcpy(result_gpu, y_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);

    bool flag = false;
#pragma omp parallel for
    for (int i = 0; i < size; i++)
        if (y[i] != result_gpu[i]) {
            flag = true;
            break;
        }

    /*if (flag == true)
    cout << "Not equal" << endl;
    else
    cout << "Equal" << endl;*/

    cout << "time_cpu = " << time_cpu << endl
        << "time_cpu_omp = " << time_cpu_omp << endl
        << "time_gpu = " << time_gpu << endl;

    system("pause");

    delete[]x; delete[]y;
    cudaFree(x_gpu); cudaFree(y_gpu);

    return 0;
}
