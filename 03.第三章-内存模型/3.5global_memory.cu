#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
__device__ float factor = 3.2;

__global__ void globalMemory(float *out)
{
    printf("device global memory:%.2f\n", factor);
    *out = factor;
}

int main(int argc, char **argv)
{
    int nDeviceNumber = 0;
    int error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != 0 || nDeviceNumber == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    // set up device
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if(error != 0)
    {
        printf("fail to set GPU 0 for computing\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for computing\n");
    }
    dim3 block(1, 1);
    dim3 grid(1,1);
    float *d_A;
    float h_A;

    cudaMalloc((void **)&d_A, sizeof(float));
    globalMemory<<<grid, block>>>(d_A);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_A, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Host memory:%.2f\n", h_A);
    cudaFree(d_A);
    cudaDeviceReset();
    return (0);
}
