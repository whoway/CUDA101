#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>


__global__ void cas_kernel(int* out, int x, int value)
{
    int old = atomicCAS(out, x, value);
    printf("original data is, %d\n", old);
}


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
    int *d_out, h_out;

    cudaMalloc((void **)&d_out, sizeof(int));
    cudaMemset(d_out, 0, sizeof(int));
    cas_kernel<<<1, 1>>>(d_out, 0, 5);
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("out is replaced, value=%d\n", h_out);
    cudaFree(d_out);
    cudaDeviceReset();

    return 0;
}
