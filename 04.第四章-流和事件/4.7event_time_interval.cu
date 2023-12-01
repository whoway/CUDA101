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
    float *pinned_A;
    size_t nBytes = nElem * sizeof(float);
    ErrorCheck(cudaHostAlloc((void**)&pinned_A, nBytes, cudaHostAllocDefault), __FILE__, __LINE__);
    initialData(pinned_A, nElem);

    // allocate gpu global memory
    float *d_A;
    cudaMalloc((float**)&d_A, nBytes);

    // transfer data from host to device
    cudaStream_t data_stream;
    cudaStreamCreate(&data_stream);

    cudaEvent_t begin_event;
    ErrorCheck(cudaEventCreate(&begin_event), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(begin_event, data_stream),__FILE__, __LINE__);
    cudaMemcpyAsync(d_A, pinned_A, nBytes, cudaMemcpyHostToDevice, data_stream);

    cudaEvent_t end_event;
    ErrorCheck(cudaEventCreate(&end_event), __FILE__, __LINE__);
    ErrorCheck(cudaEventRecord(end_event, data_stream),__FILE__, __LINE__);
    cudaEventSynchronize(end_event);
    float timeElapse = 0.0;
    cudaEventElapsedTime(&timeElapse, begin_event, end_event);
    printf("time elapse for data copy from host to device is :%.2f(ms)\n", timeElapse);

    cudaFreeHost(pinned_A);
    cudaStreamDestroy(data_stream);
    cudaEventDestroy(begin_event);
    cudaEventDestroy(end_event);
    cudaDeviceReset();

    return 0;
}
