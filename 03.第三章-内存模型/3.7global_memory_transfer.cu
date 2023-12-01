#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
__device__ float factor = 0;

__global__ void globalMemory()
{
    printf("device global memory:%.2f\n", factor);
    factor += 1.2;
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
    float h_A = 3.6;
    ErrorCheck(cudaMemcpyToSymbol(factor, &h_A, sizeof(float), 0, cudaMemcpyHostToDevice),__FILE__, __LINE__);
    globalMemory<<<grid, block>>>();
    cudaDeviceSynchronize();
    ErrorCheck(cudaMemcpyFromSymbol(&h_A, factor, sizeof(float), 0, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    printf("cudaMemcpyFromSymbol result is:%.2f\n", h_A);

    //get global address
    float *pd_A;
    ErrorCheck(cudaGetSymbolAddress((void**)&pd_A, factor), __FILE__, __LINE__);
    cudaMemcpy(&h_A, pd_A, sizeof(float), cudaMemcpyDeviceToHost);
    printf("cudaGetSymbolAddress result is:%.2f\n", h_A);

    cudaDeviceReset();
    return (0);
}
