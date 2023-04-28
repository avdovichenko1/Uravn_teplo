#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char *argv[]) {
    int max_num_iter;
    int raz;
    double max_toch;
    if (argc < 4){
        printf("Not enough args");
        exit(1);
    }
    else
    {
        max_num_iter = atoi(argv[1]); // количество итераций
        if (max_num_iter == 0){
            printf("incorrect first param");
            exit(1);
        }
        max_toch = atof(argv[2]); // точность
        if (max_toch == 0){
            printf("incorrect second param");
            exit(1);
        }
        raz = atoi(argv[3]); // размер сетки
        if (raz == 0){
            printf("incorrect third param");
            exit(1);
        }
    }
   
    clock_t a=clock();

    double* arr_pred = (double*)calloc(raz * raz, sizeof(double));
    double* arr_new = (double*)calloc(raz * raz, sizeof(double));

    arr_pred[0] = 10;
    arr_pred[raz-1] = 20;
    arr_pred[raz * (raz - 1) +raz - 1] = 30;
    arr_pred[raz * (raz-1)] = 20;

    int num_iter = 0;
    double error = 1 + max_toch;
    double shag = (10.0 / (raz - 1));
    
// выделение памяти на устройстве и копирование данных из памяти хоста в память устройства
#pragma acc enter data create(arr_pred[0:raz*raz], arr_new[0:raz*raz]) copyin(raz, shag)
    
// ядро, выполняющее циклическое заполнение массива arr_pred
#pragma acc kernels
    {
#pragma acc loop independent
        for (int j = 0; j < raz; j++) {
            arr_pred[j] = 10 + j * (10.0 / (raz - 1));
            arr_pred[j * raz] = 10 + j * (10.0 / (raz - 1));
            arr_pred[(raz - 1) * raz + j] = 20 + j * (10.0 / (raz - 1));
            arr_pred[j * raz + (raz - 1)] = 20 + j * (10.0 / (raz - 1));
        }
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    double *dop;

    while (max_num_iter > num_iter && max_toch < error) {
        num_iter++;
        if (num_iter % 100 == 0 || num_iter == 1) {
            
//объявляется область данных, которые находятся на устройстве
#pragma acc data present(arr_pred[0:raz*raz], arr_new[0:raz*raz])
            
//async(1) указывает, что выполнение этого блока должно начаться после выполнения предыдущего блока
#pragma acc kernels async(1)
            {
//директива указывает на то, что следующий цикл for может быть распараллелен, collapse(2) отвечает за то, что оба вложенных цикла могут быть распараллелены
#pragma acc loop independent collapse(2)
                for (int i = 1; i < raz - 1; i++) {
                    for (int j = 1; j < raz - 1; j++) {
                        arr_pred[i * raz + j] =0.25 * (arr_new[(i + 1) * raz + j] + arr_new[(i - 1) * raz + j] + arr_new[i * raz + j - 1] + arr_new[i * raz + j + 1]);
                    }
                }
            }
           
            int max_id = 0; //хранение индекса максимального элемента массива arr_new
            const double alpha = -1;
            
// останавливает выполнение программы, пока не завершатся все ядра, запущенные с использованием async()
#pragma acc wait
            
//директива определяет, что данные массивов находятся и на устройстве, и на хосте, и могут использоваться и изменяться на обоих уровнях
#pragma acc host_data use_device(arr_pred, arr_new)
            {
                
                cublasDaxpy(handle, raz * raz, &alpha, arr_pred, 1, arr_new, 1); // функция вычисляет значение -1 * arr_pred + arr_new и сохраняет результат в arr_new
                cublasIdamax(handle, raz * raz, arr_new, 1, &max_id); //находит индекс максимального элемента массива arr_new
            }
//копирует один элемент массива arr_new с индексом max_id-1 с устройства на хост
#pragma acc update self(arr_new[max_id-1:1])
            
            error = fabs(arr_new[max_id - 1]);
#pragma acc host_data use_device(arr_pred, arr_new)
            cublasDcopy(handle, raz * raz, arr_pred, 1, arr_new, 1); //функция копирует содержимое массива 
            
//указывает, что все ранее запланированные ядра и данные, связанные с ускорителем, должны завершить свою работу, прежде чем продолжить выполнение кода на хост-процессоре
#pragma acc wait(1)
            printf("Номер итерации: %d, ошибка: %0.8lf\n", num_iter, error);
            

        }
        else {
#pragma acc data present(arr_pred[0:raz*raz], arr_new[0:raz*raz])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < raz - 1; i++) {
                    for (int j = 1; j < raz - 1; j++) {
                        arr_pred[i * raz + j] =0.25 * (arr_new[(i + 1) * raz + j] + arr_new[(i - 1) * raz + j] + arr_new[i * raz + j - 1] + arr_new[i * raz + j + 1]);
                    }
                }
            }
        }
        dop = arr_new;
        arr_new = arr_pred;
        arr_pred = dop;
       
    }

     //вывод сетки размером 15*15
        printf("\n%d grid:\n", num_iter);
        if (raz==15){
                for (int i = 0; i < raz; i++) {
                    for (int j = 0; j < raz; j++) {
                        printf("%0.2lf ", arr_new[i * raz + j]);
                    }   
                printf("\n");
                }
            }

    printf("Final result: %d, %0.6lf\n", num_iter, error);
    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC; // переводит в секунды 
    printf("%.25f время в секундах", d);
    cublasDestroy(handle); //освобождает ресурсы, связанные с объектом handle
    free(arr_pred); 
    free(arr_new);
    return 0;
}
