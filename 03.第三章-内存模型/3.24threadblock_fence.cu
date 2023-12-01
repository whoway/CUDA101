#include <cuda_runtime.h>
#include "../common/common.h"
#include <stdio.h>

__global__ void thread_block_fence()
{
    __shared__ float shared;
    shared = 0;

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if((id/32) == 0 && id == 0)
    {
        shared = 5.0;
    }
    else if((id/32) != 0 && id == 32)
    {
        shared = 6.0;
    }
    __threadfence_block();
    printf("access local shared in thread_fence, shared=%.2f, blockIdx=%d, threadIdx=%d, threadId=%d\n",
            shared, blockIdx.x, threadIdx.x, id);
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

    //calculate on GPU
    dim3 block (32);
    dim3 grid  (2);

    thread_block_fence<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
