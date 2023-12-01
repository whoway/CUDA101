#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    //get GPU decice count
    int nDeviceNumber = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNumber == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        return -1;
    }
    // set up device
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing\n");
        return -1;
    }
    else
    {
        printf("set GPU 0 for computing\n");
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if(deviceProp.concurrentKernels)
    {
        printf("cocurrent kernel is supported on this GPU, begin to execute kernel_1\n");

    }
    else
    {
        printf("cocurrent kernel is not supported on this GPU\n");
    }

    cudaDeviceReset();
    return 0;
}
