#include <cstdlib>
#include <cstdio>
#include <malloc.h>
#include <time.h>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#define max(x, y) ((x) > (y) ? (x) : (y))


__global__ void updateTemperature(const double* arr_pred, double* arr_new, int N){
    //Индекс j вычисляется как произведение номера блока по вертикальной оси (blockIdx.y) на размер блока по вертикальной оси
    // (blockDim.y),плюс номер потока внутри блока по вертикальной оси (threadIdx.y), что позволяет потокам различных блоков
    // и потокам внутри одного блока работать с различными строками массива данных, i - аналогично.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

//проверяется, что индексы j и i находятся внутри диапазона от 1 до N + 1, чтобы исключить обработку граничных элементов массива.
    if (j > 0 && j < N + 1)
        if (i > 0 && i < N + 1)
            //новое значение элемента массива arr_new[j * (N + 2) + i] вычисляется на основе предыдущего состояния массива
            // arr_pred,используя формулу теплопроводности
            arr_new[j * (N + 2) + i] = 0.25 * (arr_pred[(j + 1) * (N + 2) + i] + arr_pred[(j - 1) * (N + 2) + i] +
                                               arr_pred[j * (N + 2) + i - 1] + arr_pred[j * (N + 2) + i + 1]);
}

__global__ void updateError(const double* arr_pred, double* arr_new, int N, double tol, double* tol1){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j > 0 && j < N + 1)
        if (i > 0 && i < N + 1) {
            arr_new[j * (N + 2) + i] = 0.25 * (arr_pred[(j + 1) * (N + 2) + i] + arr_pred[(j - 1) * (N + 2) + i] + arr_pred[j * (N + 2) + i - 1] + arr_pred[j * (N + 2) + i + 1]);
            //Вычисление значения погрешности между новым значением элемента и соответствующим предыдущим значением элемента
            tol1[i * j - 1] = max(arr_new[j * (N + 2) + i] - arr_pred[j * (N + 2) + i], tol);
        };
}


__global__ void reduceError(double* tol1, double* tolbl, int N){
    int thread_id = threadIdx.x; // индекс текущего потока внутри блока
    int global_size = blockDim.x * gridDim.x;  //вычисляет общее количество потоков на сетке ( путем умножения количества
    // потоков в блоке (blockDim.x) на количество блоков в сетке (gridDim.x))
    int global_id = blockDim.x * blockIdx.x + threadIdx.x; //вычисляет глобальный индекс текущего потока (включает в
    // себя индекс блока и индекс потока внутри блока)
    double tol = tol1[0];
    int i  = global_id;
    while(i < N){
        tol = max(tol, tol1[i]);
        i += global_size;
    }
    extern __shared__ double shared_array[]; //объявляется внешняя область памяти
    shared_array[thread_id] = tol;
    __syncthreads(); //выполняется синхронизация потоков, чтобы убедиться, что все потоки закончили запись в общую память
    int size = blockDim.x / 2;
    while (size > 0){
        if (size > thread_id)
            shared_array[thread_id] = max(shared_array[thread_id + size], shared_array[thread_id]);
        __syncthreads();
        size /= 2;
    }

    if (thread_id == 0)
        tolbl[blockIdx.x] = shared_array[0]; // значение максимальной ошибки сохраняется только в одном потоке с индексом 0 внутри блока
}


int main(int argc, char* argv[]) {
    clock_t a=clock();
    int size;
    double tol;
    int iter_max;
    if (argc < 4){
        printf("Неправильное количество аргументов");
        exit(1);
    }
    tol = strtod(argv[1], NULL);
    if (tol <= 0){
        printf("Ограничение точности должно превышать 0");
        exit(1);
    }
    size = atoi(argv[2]);
    if (size <= 0){
        printf("Размер матриццы должен быть больше 0");
        exit(1);
    }
    iter_max = atoi(argv[3]);
    if (iter_max <= 0){
        printf("Максимальное количество итераци должно быть больше 0");
        exit(1);
    }

    double *arr_pred = (double*)malloc((size + 2) * (size + 2) * sizeof(double));
    double *arr_new = (double*)malloc((size + 2) * (size + 2) * sizeof(double));

    int num_iter = 0;
    double error = 1.0;
    double shag = 10.0 / (size + 2);

    int size_pot=32; // количество потоков
    dim3 Block_size(size_pot, size_pot, 1); //размер блока и определение количества потоков в каждом блоке, 1 блок - 1024 потока
    dim3 Grid_Size((size + 33)/size_pot, (size + 33)/size_pot, 1); // определяет количество блоков в сетке, рассчитывается
    // на основе размера матрицы size и используется для распределения блоков на сетке таким образом, чтобы охватить
    // всю матрицу и иметь достаточное количество блоков для выполнения параллельных вычислений на GPU

    double* arr_pred_gp, *arr_new_gp;
    cudaMalloc((void**)&arr_pred_gp, sizeof(double ) * (size + 2) * (size + 2));
    cudaMalloc((void**)&arr_new_gp, sizeof(double ) * (size + 2) * (size + 2));

    int len_host = size + 2;

    for (int i = 0; i < size + 2; i++){
        arr_pred[i * len_host] = 10 + shag * i;
        arr_pred[i] = 10 + shag * i;
        arr_pred[len_host * (size + 1) + i] = 20 + shag * i;
        arr_pred[len_host * i + size + 1] = 20 + shag * i;

        arr_new[len_host * i] = arr_pred[i * len_host];
        arr_new[i] = arr_pred[i];
        arr_new[len_host * (size + 1) + i] = arr_pred[len_host * (size + 1) + i];
        arr_new[len_host * i + size + 1] = arr_pred[len_host * i + size + 1];
    }

    double* d_ptr;
    cudaMalloc((void **)(&d_ptr), sizeof(double)); // временное хранение указателя на GPU

    double* mas_error;
    cudaMalloc(&mas_error, sizeof(double) * (size * size)); // выделение памяти для GPU
    // копирование данных из хоста на устройство
    cudaMemcpy(arr_pred_gp, arr_pred, sizeof(double) * (size + 2) * (size + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(arr_new_gp, arr_new, sizeof(double) * (size + 2) * (size + 2), cudaMemcpyHostToDevice);


    dim3 Error_block(1024,1,1); //  размер блока (block size) для запуска ядра на GPU
    dim3 Error_grid(ceil((size * size)/(float)Error_block.x), 1, 1); // размер сетки для запуска ядра на GPU

    double* itog; // указатель на выделенную память на GPU для хранения результата вычислений ядра reduceError
    cudaMalloc(&itog, sizeof(double) * Error_grid.x);

    cudaDeviceSynchronize(); // для синхронизации выполнения всех операций на устройстве CUDA

    while ((error > tol) && (num_iter < iter_max)){
        num_iter++;
        if ((num_iter % 100 == 0) || (num_iter == 1)){
            error = 0.0;
            updateError<<<Grid_Size, Block_size>>>(arr_pred_gp, arr_new_gp, size, error, mas_error); // ядро обновляет значения массивов arr_pred_gp и arr_new_gp
            reduceError<<<Error_grid, Error_block, (Error_block.x) * sizeof(double)>>>(mas_error, itog, size * size);
            reduceError<<<1, Error_block, (Error_block.x) * sizeof(double)>>>(itog, mas_error, Error_grid.x);
            cudaMemcpy(&error, &mas_error[0], sizeof(double), cudaMemcpyDeviceToHost);

            d_ptr = arr_pred_gp;
            arr_pred_gp = arr_new_gp;
            arr_new_gp = d_ptr;

            printf("%d : %lf\n", num_iter, error);
            fflush(stdout); //  проверить, что все данные, которые были записаны в буфер вывода с помощью функции printf(), записались
        }
        else {
            updateTemperature<<<Grid_Size, Block_size>>>(arr_pred_gp, arr_new_gp, size);
            d_ptr = arr_pred_gp;
            arr_pred_gp = arr_new_gp;
            arr_new_gp = d_ptr;
        }
        updateTemperature<<<Grid_Size, Block_size>>>(arr_pred_gp, arr_new_gp, size);
    }
    printf("Финальные результаты: %d, %0.6lf\n", num_iter, error);
    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC; // переводит в секунды
    printf("%.25f время в секундах", d);
    return 0;
}
