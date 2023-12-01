#include <cuda_runtime.h>
#include "../common/common.h"
#include <stdio.h>

__global__ void thread_barrier()
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float shared;
    shared = 0.0;
    if((id / 32) == 0)
    {
        __syncthreads();
        shared = 5.0;
    }
    else
    {
       while(shared == 0.0)
       {

       }
    }
    printf("access local shared in thread_barrier, shared=%.2f, blockIdx=%d, threadIdx=%d, threadId=%d\n",
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
    dim3 block (64);
    dim3 grid  (1);
    thread_barrier<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
