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
    long count = 0;
    while(count < 9999)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < N) C[i] = A[i] + B[i];
        count += 1;
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
    //get the supported priority on this device
    int lowPriority = 0;
    int highPriority = 0;
    cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority);
    printf("Priority Range is from %d to %d\n", lowPriority, highPriority);
    int nElem = 1 << 24;

    // malloc host pinned memory
    float *pinned_A, *pinned_B, *h_C;
    size_t nBytes = nElem * sizeof(float);
    ErrorCheck(cudaHostAlloc((void**)&pinned_A, nBytes, cudaHostAllocDefault), __FILE__, __LINE__);
    ErrorCheck(cudaHostAlloc((void**)&pinned_B, nBytes, cudaHostAllocDefault), __FILE__, __LINE__);
    h_C = (float*)malloc(nBytes);

    initialData(pinned_A, nElem);
    initialData(pinned_B, nElem);

    // allocate gpu global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaStream_t data_stream;
    cudaStreamCreate(&data_stream);

    cudaMemcpyAsync(d_A, pinned_A, nBytes, cudaMemcpyHostToDevice, data_stream);
    cudaMemcpyAsync(d_B, pinned_B, nBytes, cudaMemcpyHostToDevice, data_stream);
    cudaEvent_t cp_evt;
    ErrorCheck(cudaEventCreate(&cp_evt), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(cp_evt, data_stream),__FILE__, __LINE__);

    //wait for data copy to complete
    cudaEventSynchronize(cp_evt);

    //calculate on GPU
    dim3 block (1024);
    dim3 grid  ((nElem + block.x -1) / block.x);
    cudaStream_t kernel_stream;
    cudaStreamCreateWithPriority(&kernel_stream, cudaStreamDefault, highPriority);
    sumArraysOnGPU<<<grid, block, 0, kernel_stream>>>(d_A, d_B, d_C, nElem);

    cudaEvent_t kernel_end_event;
    ErrorCheck(cudaEventCreate(&kernel_end_event), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(kernel_end_event, kernel_stream),__FILE__, __LINE__);
    cudaEventSynchronize(kernel_end_event);
    printf("return from Kernel function\n");

    cudaMemcpyAsync(h_C, d_C, nBytes, cudaMemcpyDeviceToHost, data_stream);

    cudaEvent_t cp_evt_out;
    ErrorCheck(cudaEventCreate(&cp_evt_out), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(cp_evt_out, data_stream),__FILE__, __LINE__);
    cudaEventSynchronize(cp_evt_out);
    printf("return from cudaMemcpyAsync to host\n");

    for (int i = nElem - 1; i >  nElem - 50; i--)
    {
        printf("ElemIdx=%d, matrix_A:%.2f, matrix_B:%.2f, result=%.2f\n", i, pinned_A[i],pinned_B[i],h_C[i] );
    }
    cudaFreeHost(pinned_A);
    cudaFreeHost(pinned_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(kernel_stream);
    cudaStreamDestroy(data_stream);
    cudaEventDestroy(cp_evt);
    cudaEventDestroy(cp_evt_out);
    cudaEventDestroy(kernel_end_event);
    cudaDeviceReset();

    return 0;
}
