#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));
    printf("Matrix is: ");
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    return;
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N, const int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if (k < N) C[k] = A[i] + B[i];
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
    int nElem = 1 << 12;

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
    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(gpuRef,  0, nBytes);
    //allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    if(d_A == NULL || d_B == NULL || d_C == NULL){
        printf("fail to allocate memory for GPU\n");
        free(h_A);
        free(h_B);
        free(gpuRef);
        return -1;
    }
    else
    {
        printf("successfully allocate memory for GPU\n");
    }

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice);

    //calculate on GPU
    dim3 block (256);
    dim3 grid  ((nElem + block.x - 1) / block.x, 1);
    int offset = 3;
    if(2 == argc)
    {
        offset = atoi(argv[1]);
    }
    double dTime_Begin = GetCPUSecond();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    double dTime_End = GetCPUSecond();
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);


    for (int i = nElem - 1; i > nElem - 50; i--)
    {
        printf("idx=%d, matrix_A:%.2f, matrix_B:%.2f, result=%.2f\n", i+1, h_A[i],h_B[i],gpuRef[i] );
    }

    printf("Element Size:%d, threadConfig:<<< %d, %d >>>, offset:%d, Matrix add time Elapse is:%.5f\n", nElem,     grid.x, block.x, offset, dTime_End - dTime_Begin);
    free(h_A);
    free(h_B);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();

    return 0;
}
