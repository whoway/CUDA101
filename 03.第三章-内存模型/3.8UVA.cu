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
    float *d_mem = NULL;
    ErrorCheck(cudaMalloc((void**)&d_mem, sizeof(float)), __FILE__, __LINE__);
    cudaPointerAttributes pt_Attribute;
    ErrorCheck(cudaPointerGetAttributes(&pt_Attribute, d_mem), __FILE__, __LINE__);

    printf("pointer Attribute:device=%d, devicePointer=%p, type=%d\n",
            pt_Attribute.device, pt_Attribute.devicePointer, pt_Attribute.type);
    cudaFree(d_mem);
    cudaDeviceReset();

    return 0;
}
