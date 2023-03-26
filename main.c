#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int main(int argc, char *argv[]) {
    int max_num_iter = atoi(argv[1]); // количество итераций
    double max_toch = atof(argv[2]); // точность
    int raz = atoi(argv[3]); // размер сетки
    clock_t a=clock();
    double *arr_pred = (double *)calloc(raz*raz, sizeof(double));
    double *arr_new = (double *)calloc(raz*raz, sizeof(double));

    arr_pred[0] = 10;
    arr_pred[raz - 1] = 20;
    arr_pred[(raz - 1)*raz + (raz - 1)] = 20;
    arr_pred[(raz - 1)*raz] = 30;

#pragma acc parallel loop
    for (int j = 1; j < raz; j++) {
        arr_pred[j] = (arr_pred[raz - 1] - arr_pred[0]) / (raz - 1) + arr_pred[j - 1];
        arr_pred[(raz - 1)*raz + j] = (arr_pred[(raz - 1)*raz + (raz - 1)] - arr_pred[(raz - 1)*raz]) / (raz - 1) + arr_pred[(raz - 1)*raz + j - 1];
        arr_pred[j*raz] = (arr_pred[(raz - 1)*raz] - arr_pred[0]) / (raz - 1) + arr_pred[(j - 1)*raz];
        arr_pred[j*raz + raz - 1] = (arr_pred[(raz - 1)*raz + (raz - 1)] - arr_pred[raz - 1]) / (raz - 1) + arr_pred[(j - 1)*raz + raz - 1];
    }

    int num_iter = 0;
    double error = max_toch + 1;

#pragma acc data copy(arr_pred[:raz*raz]) create(arr_new[:raz*raz])
    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        while (max_num_iter > num_iter && max_toch < error) {
            error = 0;
#pragma acc parallel loop reduction(max:error)
            for (int j = 1; j < raz - 1; j++) {
#pragma acc loop reduction(max:error)
                for (int i = 1; i < raz - 1; i++) {
                    arr_new[j*raz + i] = (arr_pred[(j - 1)*raz + i] + arr_pred[j*raz + i - 1] + arr_pred[j*raz + i + 1] + arr_pred[(j + 1)*raz + i]) * 0.25;
                    error = fmax(fabs(arr_pred[j*raz + i] - arr_new[j*raz + i]), error);
                }
            }
            cublasDcopy(handle, raz * raz, arr_new, 1, arr_pred, 1);
#pragma acc parallel loop
            for (int j = 1; j < raz - 1; j++) {
#pragma acc loop
                for (int i = 1; i < raz - 1; i++) {
                    arr_pred[j*raz + i] = arr_new[j*raz + i];
                }
            }
            if (num_iter % 10 == 0) {
                printf("Номер итерации: %d, ошибка: %0.8lf\n", num_iter, error);
            }
            num_iter++;
        }
        cublasDestroy(handle);
    }
    printf("Итог программы: %d, %0.6lf\n", num_iter, error);
    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC;
    printf("%.25f время в секундах", d);
    free(arr_pred);
    free(arr_new);
    return 0;
}
