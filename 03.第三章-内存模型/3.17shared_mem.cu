#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>

//declear statical global shared memory variable
__shared__ float g_shared;

__global__ void kernel_1()
{
    __shared__ float k1_shared;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(blockIdx.x == 0 && id == 0 )
    {
        k1_shared = 5.0;
    }
    if(blockIdx.x == 1 && id == 16)
    {
        k1_shared = 6.0;
    }
    __syncthreads();
    printf("access local shared in kernel_1, k1_shared=%.2f, blockIdx=%d, threadIdx=%d, threadId=%d\n",
            k1_shared, blockIdx.x, threadIdx.x, id);
}


__global__ void kernel_2()
{
    g_shared = 0.0;
    printf("access global shared in kernel_2, g_shared=%.2f\n", g_shared);
    //printf("access local_shared in kernel_2, k1_shared=%.2f\n", k1_shared);
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
    dim3 block (16);
    dim3 grid  (2);

    kernel_1<<<grid, block>>>();
    kernel_2<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
