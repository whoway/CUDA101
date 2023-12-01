#include "common/common.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    float* gpuMemory = NULL;
    ErrorCheck(cudaMalloc(&gpuMemory, sizeof(float)), __FILE__, __LINE__);
    ErrorCheck(cudaFree(gpuMemory), __FILE__, __LINE__);
    ErrorCheck(cudaFree(gpuMemory), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 1;
}


