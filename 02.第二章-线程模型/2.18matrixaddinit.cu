#include <cuda_runtime.h>
#include "../common/common.h"
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
        printf("%.2f ",ip[i] );
    }
    printf("\n");
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
    int nElem = 16;

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
    free(h_A);
    free(h_B);
    free(gpuRef);
    return 0;
}
