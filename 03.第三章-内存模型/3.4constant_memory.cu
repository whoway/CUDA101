#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__constant__ float factor;

__global__ void constantMemory()
{
    printf("Get constant memory:%.2f\n", factor);
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
    dim3 block(8, 1);
    dim3 grid(1,1);
    float h_factor = 2.3;
    ErrorCheck(cudaMemcpyToSymbol(factor, &h_factor, sizeof(float), 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    constantMemory<<<grid, block>>>();
    cudaDeviceSynchronize();
    // reset device
    cudaDeviceReset();
    return (0);
}
