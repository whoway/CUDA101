#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void zerocopyMemory(float* input)
{
    printf("GPU zero-copy memory:%.2f\n", *input);
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
    float *h_zerocpyMem = NULL;
    ErrorCheck(cudaHostAlloc((float**)&h_zerocpyMem, sizeof(float), cudaHostAllocDefault), __FILE__, __LINE__);

    *h_zerocpyMem = 4.8;
    printf("CPU zero-copy memory:%.2f\n", *h_zerocpyMem);
    zerocopyMemory<<<grid, block>>>(h_zerocpyMem);
    cudaDeviceSynchronize();
    cudaFreeHost(h_zerocpyMem);


    cudaDeviceReset();
    return (0);
}
