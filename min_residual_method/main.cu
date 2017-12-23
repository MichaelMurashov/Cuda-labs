#include <cstdlib>
#include <iostream>
#include <string>
#include <time.h>

#include "device_launch_parameters.h"
#include "cublas.h"
#include "curand.h"

#define NONE_TRANS 'N'
#define BLOCK_SIZE 32

__global__
void makePositiveOrientire( const int n, float* matrix ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n )
        matrix[ i * n + i ] += n;
}

bool isMethodSuccess( const int iter, const float error, const int nIters, const float epsilon ) {
    return nIters <= iter || error <= epsilon;
}

int main( int argc, char** argv ) {
    const int    dim     = 2048;
    const int    nIters  = 100;
    const float  epsilon = 1e-5;

    cublasInit();

    curandGenerator_t generator;
    curandCreateGenerator( &generator, CURAND_RNG_PSEUDO_DEFAULT );

    curandSetPseudoRandomGeneratorSeed( generator, 2000 );

    float *devA, *devX, *devB;

    cublasAlloc( dim*dim, sizeof(float), (void**) &devA );
    cublasAlloc( dim*1  , sizeof(float), (void**) &devX );
    cublasAlloc( dim*1  , sizeof(float), (void**) &devB );

    curandGenerateUniform( generator, devA, dim*dim );
    curandGenerateUniform( generator, devB, dim*1   );

    makePositiveOrientire <<< ( dim + BLOCK_SIZE - 1 ) / BLOCK_SIZE, BLOCK_SIZE >>> ( dim, devA );
    cudaDeviceSynchronize();

    cudaMemcpy((void*) devX, (const void*) devB, sizeof( float )*dim, cudaMemcpyDeviceToDevice);

    int    iter  = 0;
    float  error = 0.0f;
    float  *devR, *devE;

    cublasAlloc( dim*1, sizeof( float ), (void**) &devR );
    cublasAlloc( dim*1, sizeof( float ), (void**) &devE );

    clock_t start = clock();

    do {
        /* вычисления */

        /* r = A*x */
        cublasSgemv(
                NONE_TRANS, dim, dim,
                1.0f, (const float*) devA, dim, (const float*) devX, 1, 0.0f, devR, 1
        );

        /* r = r - b */
        cublasSaxpy( dim, -1.0f, (const float*) devB, 1, devR, 1 );

        /* e = A*r */
        cublasSgemv(
                NONE_TRANS, dim, dim,
                1.0f, (const float*) devA, dim, (const float*) devR, 1, 0.0f, devE, 1
        );

        float t  = cublasSdot( dim, (const float*) devE, 1, (const float*) devR, 1 );
        t /= cublasSdot( dim, (const float*) devE, 1, (const float*) devE, 1 );

        /* x = x - t*r */
        cublasSaxpy( dim, -t, (const float*) devR, 1, devX, 1 );

        /* проверка */

        /* e = A*x */
        cublasSgemv(
                NONE_TRANS, dim, dim,
                1.0f, (const float*) devA, dim, (const float*) devX, 1, 0.0f, devE, 1
        );

        /* e = e - b */
        cublasSaxpy( dim, -1, (const float*) devB, 1, devE, 1 );

        /* error = ||e|| */
        error = cublasSnrm2( dim, (const float*) devE, 1 );
    } while ( !isMethodSuccess( ++iter, error, nIters, epsilon ) );

    clock_t finish = clock();
    float time = (float)(finish - start)/CLOCKS_PER_SEC;

    std::cout << "time: " <<  time << std::endl
        << "Accuracy of method: " << error << std::endl
        << "Iter: "<< iter << " of " << nIters << std::endl;

    cublasFree( devR ); cublasFree( devE );

    cublasFree( devA ); cublasFree( devX ); cublasFree( devB );

    cublasShutdown();
    curandDestroyGenerator( generator );

    return 0;
}
