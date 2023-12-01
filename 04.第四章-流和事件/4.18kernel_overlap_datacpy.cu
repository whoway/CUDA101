#include "common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define NSTREAM 4
#define BDIM 128

void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        for (int i = 0; i < 99999; ++i)
        {
            C[idx] = A[idx] + B[idx];
        }
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

    // set up data size of vectors
    int nElem = 1 << 18;
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *gpuRef;
    cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(gpuRef,  0, nBytes);


    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // invoke kernel at host side
    dim3 block (BDIM);
    dim3 grid  ((nElem + block.x - 1) / block.x);
    printf("> grid (%d, %d) block (%d, %d)\n", grid.x, grid.y, block.x,
            block.y);

    // grid parallel operation
    int iElem = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);
    grid.x = (iElem + block.x - 1) / block.x;

    cudaStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i)
    {
        cudaStreamCreate(&stream[i]);
    }

    cudaEventRecord(start, 0);

    for (int i = 0; i < NSTREAM; ++i)
    {
        int ioffset = i * iElem;
        cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes,
                              cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes,
                              cudaMemcpyHostToDevice, stream[i]);
        sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset],
                &d_C[ioffset], iElem);
        cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes,
                              cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float execution_time;
    cudaEventElapsedTime(&execution_time, start, stop);

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
           execution_time, (nBytes * 2e-6) / execution_time );

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(gpuRef);

    // destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // destroy streams
    for (int i = 0; i < NSTREAM; ++i)
    {
        cudaStreamDestroy(stream[i]);
    }

    cudaDeviceReset();
    return(0);
}
