#include "common/common.h"
#include <stdio.h>
#include <stdlib.h>


__global__ void selfdefined_AtomicAdd(int *address, int incr)
{
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);
    while(oldValue != guess)
    {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }

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
    int incr = 1;
    if(argc == 2)
    {
        incr = atoi(argv[1]);
    }
    printf("increment is %d\n", incr);
    int h_sharedInteger;
    int *d_sharedInteger;
    cudaMalloc((void **)&d_sharedInteger, sizeof(int));
    cudaMemset(d_sharedInteger, 0x00, sizeof(int));

    selfdefined_AtomicAdd<<<4, 5>>>(d_sharedInteger, incr);
    cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int), cudaMemcpyDeviceToHost);
    printf("4 x 5 increments led to value of %d\n", h_sharedInteger);
    cudaFree(d_sharedInteger);
    cudaDeviceReset();

    return 0;
}

