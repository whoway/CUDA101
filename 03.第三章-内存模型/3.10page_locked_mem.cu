#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void pageLockedMemory(float* input)
{
    printf("GPU page-locked memory:%.2f\n", *input);
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
    float *h_PinnedMem = NULL;
    ErrorCheck(cudaMallocHost((float**)&h_PinnedMem, sizeof(float)), __FILE__, __LINE__);

    *h_PinnedMem = 4.8;
    printf("CPU page-locked memory:%.2f\n", *h_PinnedMem);
    pageLockedMemory<<<grid, block>>>(h_PinnedMem);
    cudaDeviceSynchronize();
    cudaFreeHost(h_PinnedMem);


    cudaDeviceReset();
    return (0);
}
