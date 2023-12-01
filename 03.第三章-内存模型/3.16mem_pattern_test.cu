#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>

__global__ void readOffset(float *A, float *B, float *C, const float n, float offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if (k < n)
    {
        C[i] = A[k];
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
    int nElem = 32;
    int offset = 0;
    if(argc == 2)
    {
        offset = atoi(argv[1]);
    }
    printf("offset is %d\n", offset);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);
    if(NULL != h_A && NULL != h_B && NULL != gpuRef)
    {
        printf("allocate memory successfully\n");
    }
    else
    {
        printf("fail to allocate memory\n");
        return -1;
    }
    for(int i = 0; i < nElem; i++)
    {
        h_A[i] = i;
        h_B[i] = i + 1;
    }
    memset(gpuRef,  0, nBytes);
    //allocate GPU memory
    float *d_A, *d_B, *d_C;
    ErrorCheck(cudaMalloc((float**)&d_A, nBytes), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float**)&d_B, nBytes), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float**)&d_C, nBytes), __FILE__, __LINE__);

    // transfer data from host to device
    ErrorCheck(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //calculate on GPU
    dim3 block (10);
    dim3 grid  (1);

    double start_Time = GetCPUSecond();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    double iElaps = GetCPUSecond() - start_Time;
    ErrorCheck(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    for (int i = 1; i < nElem; i++)
    {
        gpuRef[0] += gpuRef[i];
    }
    printf("ElemSize=%d, total result=%.2f\n", nElem, gpuRef[0]);
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
