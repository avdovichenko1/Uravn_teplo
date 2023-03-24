#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char *argv[]) {
    int max_num_iter = atoi(argv[1]);
    double max_toch = atof(argv[2]);
    int raz = atoi(argv[3]);
    clock_t a=clock();
    double **arr_pred= (double **)calloc(raz, sizeof(double *));
    double **arr_new= (double **)calloc(raz, sizeof(double *));

    for (int i = 0; i < raz; i++) {
        arr_pred[i] = (double *)calloc(raz, sizeof(double ));
        arr_new[i] = (double *)calloc(raz, sizeof(double ));
    }
    arr_pred[0][0] = 10;
    arr_pred[0][raz - 1] = 20;
    arr_pred[raz - 1][raz - 1] = 20;
    arr_pred[raz - 1][0] = 30;

    // Set up cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate GPU memory
    double **d_arr_pred, **d_arr_new;
    cudaMalloc(&d_arr_pred, raz * sizeof(double *));
    cudaMalloc(&d_arr_new, raz * sizeof(double *));
    for (int i = 0; i < raz; i++) {
        cudaMalloc(&d_arr_pred[i], raz * sizeof(double));
        cudaMalloc(&d_arr_new[i], raz * sizeof(double));
    }

    // Transfer data to GPU memory
    cudaMemcpy(d_arr_pred[0], arr_pred[0], raz * raz * sizeof(double), cudaMemcpyHostToDevice);

    int num_iter = 0;
    double error = max_toch + 1;

    while(max_num_iter > num_iter && max_toch < error){
        error = 0;

        // Calculate arr_new on the GPU
        double alpha = 0.25, beta = 0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, raz-2, raz-2, raz-2, &alpha,
                    d_arr_pred[1], raz, d_arr_pred[raz-2], raz, &beta,
                    d_arr_new[1], raz);

        // Calculate error on the GPU
        double *d_error;
        cudaMalloc(&d_error, sizeof(double));
        cudaMemset(d_error, 0, sizeof(double));
        double one = 1;
        cublasDasum(handle, (raz-2) * (raz-2), d_error, 1, d_arr_pred[1]+raz+1, 1);
        cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_error);

        // Reduce error on the GPU
        cublasDasum(handle, (raz-2) * (raz-2), d_arr_new[1]+raz+1, 1, &error);

        // Update arr_pred on the GPU
        cudaMemcpy(d_arr_pred[1]+raz+1, d_arr_new[1]+raz+1, (raz-2)*(raz-2)*sizeof(double), cudaMemcpyDeviceToDevice);

        if (num_iter % 10 == 0) {
            printf("Номер итерации: %d, ошибка: %0.8lf\n", num_iter, error);
        }
        num_iter++;

    }

    printf("Итог программы: %d, %0.6lf\n", num_iter, error);
    
    cudaFree(d_arr_pred);
    cudaFree(d_arr_new);
    for (int i = 0; i < raz; i++) {
        free(arr_pred[i]);
        free(arr_new[i]);
    }
    free(arr_pred);
    free(arr_new);
    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC;
    printf("%.25f время в секундах", d);
    return 0;
}
