#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>


__global__ void kernel(float *F, double *D)
{
    float val1 = *F;
    double val2 = *D;
    printf("Device single-precision representation is %.20f\n", val1);
    printf("Device double-precision representation is %.20f\n", val2);
}

int main(int argc, char **argv)
{
    float hostF = 0.0;
    double hostD = 0.0;
    if(argc == 2)
    {
        hostF = (float)atof(argv[1]);
        hostD = (double)atof(argv[1]);
    }
    else
    {
        printf("input a float point number!\n");
        return -1;
    }
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
    float *deviceF;
    double *deviceD;


    ErrorCheck(cudaMalloc((void **)&deviceF, sizeof(float)), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((void **)&deviceD, sizeof(double)), __FILE__, __LINE__);

    ErrorCheck(cudaMemcpy(deviceF, &hostF, sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(deviceD, &hostD, sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    printf("Host single-precision representation is %.20f\n",  hostF);
    printf("Host double-precision representation is %.20f\n",  hostD);


    kernel<<<1, 1>>>(deviceF, deviceD);
    cudaDeviceSynchronize();
    cudaFree(deviceF);
    cudaFree(deviceD);
    cudaDeviceReset();
    return 0;
}

