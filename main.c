#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int max_num_iter = atoi(argv[1]); // count of iterations
    double max_toch = atof(argv[2]); // tochnost
    int raz = atoi(argv[3]); // count matrix
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

#pragma acc parallel loop
    for(int j = 1; j < raz; j++){
        double temp = (arr_pred[0][raz - 1] - arr_pred[0][0]) / (raz - 1);
        arr_pred[0][j] = temp + arr_pred[0][j - 1];
        temp = (arr_pred[raz - 1][raz - 1] - arr_pred[raz - 1][0]) / (raz - 1);
        arr_pred[raz - 1][j] = temp + arr_pred[raz - 1][j - 1];
        temp = (arr_pred[raz - 1][0] - arr_pred[0][0]) / (raz - 1);
        arr_pred[j][0] = temp + arr_pred[j - 1][0];
        temp = (arr_pred[raz - 1][raz - 1] - arr_pred[0][raz - 1]) / (raz - 1);
        arr_pred[j][raz - 1] = temp + arr_pred[j - 1][raz - 1];
    }
#pragma acc data copy(arr_pred[:raz][:raz]) create(arr_new[:raz][:raz])
    {
        int tile_size = 16; // adjust this value for best performance
        double error = max_toch + 1;
        int num_iter = 0;
        while (max_num_iter > num_iter && max_toch < error) {
            error = 0;
#pragma acc parallel loop reduction(max:error)
            for (int tile_j = 1; tile_j < raz - 1; tile_j += tile_size) {
#pragma acc loop independent reduction(max:error)
                for (int tile_i = 1; tile_i < raz - 1; tile_i += tile_size) {
                    for (int j = tile_j; j < fmin(tile_j + tile_size, raz - 1); j++) {
#pragma acc loop vector independent
                        for (int i = tile_i; i < fmin(tile_i + tile_size, raz - 1); i++) {
                            arr_new[j][i] = (arr_pred[j-1][i] + arr_pred[j][i-1] + arr_pred[j][i+1] + arr_pred[j+1][i]) * 0.25;
                            error = fmax(fabs(arr_pred[j][i] - arr_new[j][i]), error);
                        }
                    }
                }
            }
#pragma acc parallel loop
            for (int j = 1; j < raz - 1; j++) {
#pragma acc loop
                for (int i = 1; i < raz - 1; i++) {
                    arr_pred[j][i] = arr_new[j][i];
                }
            }
            num_iter++;
        }
        printf("Programms result: %d, %0.6lf\n", num_iter, error);
    }
    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC;
    printf("%.25f time in sec", d);
#pragma acc parallel loop
    for (int i = 0; i < raz; i++) {
        free(arr_pred[i]);
        free(arr_new[i]);
    }
    free(arr_pred);
    free(arr_new);
    return 0;
}
