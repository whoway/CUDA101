#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void mathKernel(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}


int main(int argc, char **argv)
{
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
    // set up data size
    int size = 64;
    int blocksize = 64;

    printf("Data size %d ", size);

    // set up execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    error = ErrorCheck(cudaMalloc((float**)&d_C, nBytes), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("fail to allocate memory for GPU\n");
        return -1;
    }


    mathKernel<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();

    // free gpu memory and reset divece
    cudaFree(d_C);
    cudaDeviceReset();
    return 0;
}
