#include <cuda_runtime.h>
#include "../common/common.h"
#include <stdio.h>


void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    return;
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = threadIdx.x;
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
    // get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    // check if support mapped memory
    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        return -1   ;
    }

    // set up data size of vectors
    int nElem = 1 << 12;

    //allocate zero-copy memory
    float *h_ZeroMA = NULL;
    float *h_ZeroMB = NULL;
    float *h_ZeroMC = NULL;
    ErrorCheck(cudaHostAlloc((float**)&h_ZeroMA, nElem * sizeof(float), cudaHostAllocMapped), __FILE__, __LINE__);
    ErrorCheck(cudaHostAlloc((float**)&h_ZeroMB, nElem * sizeof(float), cudaHostAllocMapped), __FILE__, __LINE__);
    ErrorCheck(cudaHostAlloc((float**)&h_ZeroMC, nElem * sizeof(float), cudaHostAllocMapped), __FILE__, __LINE__);

    // initialize data at host side
    initialData(h_ZeroMA, nElem);
    initialData(h_ZeroMB, nElem);

    //calculate on GPU
    dim3 block (256);
    dim3 grid  (1);

    sumArraysOnGPU<<<grid, block>>>(h_ZeroMA, h_ZeroMB, h_ZeroMC, nElem);
    cudaDeviceSynchronize();

    for (int i = 0; i < 100; i++)
    {
        printf("matrix_A:%.2f, matrix_B:%.2f, result=%.2f\n", h_ZeroMA[i],h_ZeroMB[i],h_ZeroMC[i] );
    }
    cudaFreeHost(h_ZeroMA);
    cudaFreeHost(h_ZeroMB);
    cudaFreeHost(h_ZeroMC);
    cudaDeviceReset();

    return 0;
}
