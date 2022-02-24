#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



__global__ void PerElement_AtimesB(int *c, int *a, int *b)
{ 
    printf("hit");
    c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
}

int main()
{    
   
    cudaError_t cudaStatus;

    int *a, *b, *c;

    const int size = 5;

    cudaMallocManaged(&a, size * sizeof(int));
    cudaMallocManaged(&b, size * sizeof(int));
    cudaMallocManaged(&c, size * sizeof(int));

    int d = 0;

    for (int i = 0; i < size; i++)
    {
        a[i] = i + 1;
        b[i] = (i + 1) * 10;
        c[i] = 0;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }



    PerElement_AtimesB << <1, size >> > (c, a, b);

    cudaDeviceSynchronize();


    //printf("Array A = {%d", a[0]);
    //for (int i = 1; i < size; i++) {
    //    printf(",%d", a[i]);
    //}
    //printf("}\n");
    //printf("Array B = {%d", b[0]);
    //for (int i = 1; i < size; i++) {
    //    printf(",%d", b[i]);
    //}
    //printf("}\n");

    printf("Array C = {%d", c[0]);
    for (int i = 1; i < size; i++) {
        printf(",%d", c[i]);
    }
    printf("}");


    cudaFree(c);
    cudaFree(a);
    cudaFree(b);
Error:
    cudaFree(c);
    cudaFree(a);
    cudaFree(b);

    return 0;
}
