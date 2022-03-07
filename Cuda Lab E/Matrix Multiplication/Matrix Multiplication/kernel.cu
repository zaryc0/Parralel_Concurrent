#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// represents the Common Dimension between the two arrays 
__global__ void OneBlockMatrixMultiplication(int* c, int* a, int* b, int common, int cW)
{
    
    int ARow = threadIdx.y * common;
    int BColumn = threadIdx.x; 
    int CIndex = threadIdx.x + (threadIdx.y * cW);

    //printf("Thread ID: (%d,%d)\n{\n AIndex: %d\n BIndex: %d\n CIndex: %d\n}\n", threadIdx.x, threadIdx.y, ARow, BColumn, CIndex);

    for (int i = 0; i < common; i++)
    {
        c[CIndex] += (a[ARow + i]) * (b[i* cW + BColumn]);
    }
}

void ConcurrentMatrixMultiplication(int* c, int* a, int* b, int common, int W, int H)
{
    for (int x = 0; x < W; x++)
    {
        for (int y = 0; y < H; y++)
        {
            int ARow = y * common;
            int BColumn = x;
            int CIndex = x + (y * W);

            //printf("Thread ID: (%d,%d)\n{\n AIndex: %d\n BIndex: %d\n CIndex: %d\n}\n", x, y, ARow, BColumn, CIndex);

            for (int i = 0; i < common; i++)
            {
                c[CIndex] += (a[ARow + i]) * (b[i * W + BColumn]);
            }
        }
    }
}

int main()
{

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float elapsedTime;
    float time;
    cudaEventCreate(&start);

    int* a, * b, * c;
    const int heightA = 1024;
    const int widthA = heightA;
    const int heightB = heightA;
    const int widthB = heightA;
    const int arraySizeA = heightA * widthA;
    const int arraySizeB = heightB * widthB;
    const int arraySizeC = heightA * widthB;

    cudaMallocManaged(&a, arraySizeA * sizeof(int));
    cudaMallocManaged(&b, arraySizeB * sizeof(int));
    cudaMallocManaged(&c, arraySizeC * sizeof(int));

    int d = 0;

    for (int i = 0; i < arraySizeA; i++)
    {
        a[i] = 2;
    }
    for (int i = 0; i < arraySizeB; i++)
    {
        b[i] = 2;
    }
    for (int i = 0; i < arraySizeC; i++)
    {
        c[i] = 0;
    }
    cudaEventRecord(start, 0);

    OneBlockMatrixMultiplication << <1, dim3(widthB,heightA) >> > (c, a, b, widthA, widthB);
    //ConcurrentMatrixMultiplication(c, a, b, widthA, widthB, heightA);

    cudaDeviceSynchronize();
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    //printf("Array A = \n{\n  %d", a[0]);
    //for (int i = 1; i < arraySizeA; i++) {
    //    if (i % 10 == 0)
    //    {
    //        printf("\n");
    //    }
    //    printf(",%d", a[i]);
    //}
    //printf("}\n");

    //printf("Array B = \n{\n  %d", b[0]);
    //for (int i = 1; i < arraySizeB; i++) {
    //    if (i % 10 == 0)
    //    {
    //        printf("\n");
    //    }
    //    printf(",%d", b[i]);

    //}
    //printf("}\n");
    
    printf("Array C = \n{\n  %d", c[0]);
    for (int i = 1; i < arraySizeC; i++) {
        if (i % 10 == 0)
        {
            printf("\n  ");
        }
        printf(",%d", c[i]);
    }
    printf("\n}\n");

    printf("\nExecution of kernel: %fms\n", elapsedTime);
    cudaFree(c);
    cudaFree(a);
    cudaFree(b);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}
