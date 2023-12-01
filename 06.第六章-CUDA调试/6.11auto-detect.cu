#include <cuda_runtime.h>
#include "../common/common.h"
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

    //allocate GPU memory
    float *d_A;
    cudaMalloc((float**)&d_A, sizeof(float) * 10);
    cudaFree(d_A);
    cudaFree(d_A);
    cudaDeviceReset();

    return 0;
}
