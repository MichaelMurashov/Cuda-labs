#include <stdio.h>
#include <cuda_runtime_api.h>

__global__ void kernel() {
  printf("Hello, world!\n");
}

int main() {
  kernel <<< 2, 2 >>> ();
  cudaDeviceSynchronize();
  return 0;
}
