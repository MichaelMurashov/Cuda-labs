#include <time.h>
#include <omp.h>

using namespace std;

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
