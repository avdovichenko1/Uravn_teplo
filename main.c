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

#pragma acc data create(arr_pred[:raz][:raz], arr_new[:raz][:raz])
    {
        for(int i = 1; i < raz - 1; i++){
            double temp = (arr_pred[0][raz - 1] - arr_pred[0][0]) / (raz - 1);
            arr_pred[0][i] = temp + arr_pred[0][i - 1];
            temp = (arr_pred[raz - 1][raz - 1] - arr_pred[raz - 1][0]) / (raz - 1);
            arr_pred[raz - 1][i] = temp + arr_pred[raz - 1][i - 1];
            temp = (arr_pred[raz - 1][0] - arr_pred[0][0]) / (raz - 1);
            arr_pred[i][0] = temp + arr_pred[i - 1][0];
            temp = (arr_pred[raz - 1][raz - 1] - arr_pred[0][raz - 1]) / (raz - 1);
            arr_pred[i][raz - 1] = temp + arr_pred[i - 1][raz - 1];
        }

        int num_iter = 0;
        double error = max_toch + 1;

        while (max_num_iter > num_iter && max_toch < error) {
            error = 0;

#pragma acc parallel loop reduction(max:error)
            for (int i = 1; i < raz - 1; i++) {
#pragma acc loop reduction(max:error)
                for (int j = 1; j < raz - 1; j++) {
                    arr_new[i][j] =
                            (arr_pred[i - 1][j] + arr_pred[i][j - 1] + arr_pred[i][j + 1] + arr_pred[i + 1][j]) *
                            0.25;
                    error = fmax(fabs(arr_pred[i][j] - arr_new[i][j]), error);
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
