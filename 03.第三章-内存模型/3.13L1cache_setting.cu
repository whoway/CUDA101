#include <cuda_runtime.h>
#include "../common/common.h"
#include <stdio.h>

__global__ void sumArraysOnGPU(int *A, int *B, int *C, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) C[i] = A[i] + B[i];
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

    //check whether L1 cache is supported
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if(deviceProp.globalL1CacheSupported)
    {
        printf("Global L1 cache is supported, %d!\n", deviceProp.globalL1CacheSupported);
    }
    else
    {
        printf("Global L1 cache is not supported, %d!\n", deviceProp.globalL1CacheSupported);
    }
    

    // set up data size of vectors
    int nElem = 1 << 24;

    // malloc host memory
    size_t nBytes = nElem * sizeof(int);

    int *h_A, *h_B, *gpuRef;
    h_A     = (int *)malloc(nBytes);
    h_B     = (int *)malloc(nBytes);
    gpuRef  = (int *)malloc(nBytes);
    if(NULL != h_A && NULL != h_B && NULL != gpuRef)
    {
        printf("allocate memory successfully\n");
    }
    else
    {
        printf("fail to allocate memory\n");
        return -1;
    }
    // initialize data at host side
    for(int i = 0; i < nElem; i++)
    {
        h_A[i] = i;
        h_B[i] = i + 1;
    }
    memset(gpuRef,  0, nBytes);
    //allocate GPU memory
    int *d_A, *d_B, *d_C;
    ErrorCheck(cudaMalloc((int**)&d_A, nBytes), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&d_B, nBytes), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&d_C, nBytes), __FILE__, __LINE__);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice);

    //calculate on GPU
    dim3 block (1024);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    double start_Time = GetCPUSecond();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = GetCPUSecond() - start_Time;

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    for (int i = nElem - 1; i >  nElem - 50; i--)
    {
        printf("ElemSize=%d, index=%d, matrix_A:%d, matrix_B:%d, result=%d\n", nElem, i+1, h_A[i],h_B[i],gpuRef[i] );
    }
    printf("sumArraysOnGPU <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
            grid.y, block.x, block.y, iElaps);
    free(h_A);
    free(h_B);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();

    return 0;
}
