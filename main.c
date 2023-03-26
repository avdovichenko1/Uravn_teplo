#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define BLOCK_SIZE 32

__global__ void updateArr(double* arr_new, double* arr_pred, double* max_error, int raz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < raz - 1 && j < raz - 1) {
        double val = (arr_pred[(i-1)*raz+j] + arr_pred[i*raz+j-1] + arr_pred[i*raz+j+1] + arr_pred[(i+1)*raz+j]) * 0.25;
        arr_new[i*raz+j] = val;
        double error = fabs(arr_pred[i*raz+j] - val);
        atomicMax(max_error, error);
    }
}

int main(int argc, char *argv[]) {
    int max_num_iter = atoi(argv[1]); 
    double max_toch = atof(argv[2]); 
    int raz = atoi(argv[3]); 
    clock_t a=clock();

    double* arr_pred;
    double* arr_new;
    cudaMallocManaged(&arr_pred, raz * raz * sizeof(double));
    cudaMallocManaged(&arr_new, raz * raz * sizeof(double));
    cudaMemset(arr_pred, 0, raz * raz * sizeof(double));
    arr_pred[0*raz+0] = 10;
    arr_pred[0*raz+(raz-1)] = 20;
    arr_pred[(raz-1)*raz+(raz-1)] = 20;
    arr_pred[(raz-1)*raz+0] = 30;

    int num_iter = 0;
    double error = max_toch + 1;

    cublasHandle_t handle;
    cublasCreate(&handle);

    while(max_num_iter > num_iter && max_toch < error){
        error = 0;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((raz-2+BLOCK_SIZE)/BLOCK_SIZE, (raz-2+BLOCK_SIZE)/BLOCK_SIZE);
        updateArr<<<dimGrid, dimBlock>>>(arr_new, arr_pred, &error, raz);
        cudaDeviceSynchronize();
        cublasDcopy(handle, raz * raz, arr_new, 1, arr_pred, 1);
        if (num_iter % 10 == 0) {
            printf("Номер итерации: %d, ошибка: %0.8lf\n", num_iter, error);
        }
        num_iter++;
    }

    cublasDestroy(handle);
    cudaFree(arr_pred);
    cudaFree(arr_new);

    printf("Programm result: %d, %0.6lf\n", num_iter, error);
    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC;
    printf("%.25f time in sec", d);

    return 0;
}
