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
    int i = blockIdx.x; // размер строки
    int j = threadIdx.x; // столбца
    int k = blockDim.x + 1; // блока
//проверка находится ли элемент массива внутри границы
   if (j != 0 && j != k - 1)
        if (i != 0 && i != k - 1)
            arr_new[i * k + j] = 0.25 * (arr_pred[i*k+j-1] + arr_pred[(i - 1) * k + j] +
                                               arr_pred[(i+1)*k+j] + arr_pred[i * k + j + 1]);
}

__global__ void updateError(const double* arr_pred, double* arr_new, int N, double tol, double* tol1){
   //локальныt индексs элементов в блоке, в котором выполняется поток
    int i = (blockIdx.x + gridDim.y + blockIdx.y)*blockDim.x+blockDim.y+(threadIdx.x+threadIdx.y*threadIdx.x) / (gridDim.x * blockDim.x);
    int j = (blockIdx.x + gridDim.y + blockIdx.y)*blockDim.x+blockDim.y+(threadIdx.x+threadIdx.y*threadIdx.x) % (gridDim.y * blockDim.y);
// проверяется, что индексы не находятся на границах массива
    if (j != 0 && j != gridDim.x * blockDim.x -1)
        if (i != 0 && i < gridDim.y * blockDim.y - 1)
            tol1[i * (gridDim.x * blockDim.x) + j] = abs(arr_new[(i*(gridDim.x * blockDim.x) + j]-arr_pred[(i*(gridDim.x * blockDim.x) + j]); //вычисление абсолютной разности между элементами массивов
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

    double *arr_pred = (double*)malloc((size) * (size) * sizeof(double));
    double *arr_new = (double*)malloc((size) * (size) * sizeof(double));

    int num_iter = 0;
    double error = 1.0;
    double shag = 10.0 / (size-1);
    
    arr_pred[0] = 10;
    arr_pred[size-1] = 20;
    arr_pred[size * (size - 1) +size - 1] = 30;
    arr_pred[size * (size-1)] = 20;

    int size_pot=32; // количество потоков
    dim3 Block_size(size_pot, size_pot, 1); //размер блока и определение количества потоков в каждом блоке, 1 блок - 1024 потока
    dim3 Grid_Size((size + 33)/size_pot, (size + 33)/size_pot, 1); // определяет количество блоков в сетке, рассчитывается
    // на основе размера матрицы size и используется для распределения блоков на сетке таким образом, чтобы охватить
    // всю матрицу и иметь достаточное количество блоков для выполнения параллельных вычислений на GPU

    double* arr_pred_gp, *arr_new_gp;
    cudaMalloc((void**)&arr_pred_gp, sizeof(double ) * (size) * (size));
    cudaMalloc((void**)&arr_new_gp, sizeof(double ) * (size) * (size));

    int len_host = size;

    for (int i = 0; i < size-1; i++){
        arr_pred[i * len_host] = 10 + shag * i;
        arr_pred[i] = 10 + shag * i;
        arr_pred[len_host * (size - 1) + i] = 20 + shag * i;
        arr_pred[len_host * i + size - 1] = 20 + shag * i;

        arr_new[len_host * i] = arr_pred[i * len_host];
        arr_new[i] = arr_pred[i];
        arr_new[len_host * (size - 1) + i] = arr_pred[len_host * (size - 1) + i];
        arr_new[len_host * i + size - 1] = arr_pred[len_host * i + size - 1];
    }
    
    double *tempStorage = NULL; // временного хранения буфера для операции редукции на GPU
    size_t tempStorageBytes = 0;

    double* d_ptr;
    cudaMalloc((void **)(&d_ptr), sizeof(double)); // временное хранение указателя на GPU

    double* mas_error;
    cudaMalloc(&mas_error, sizeof(double) * (size * size)); // выделение памяти для GPU
    // копирование данных из хоста на устройство
    cudaMemcpy(arr_pred_gp, arr_pred, sizeof(double) * (size ) * (size), cudaMemcpyHostToDevice);
    cudaMemcpy(arr_new_gp, arr_new, sizeof(double) * (size) * (size), cudaMemcpyHostToDevice);
    
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, d_ptr, mas_error, (size) * (size)); // получение размер временного буфера для редукции
    cudaMalloc((void **)&tempStorage, tempStorageBytes); //выделение памяти для буфера


    dim3 Error_block(1024,1,1); //  размер блока (block size) для запуска ядра на GPU
    dim3 Error_grid(ceil((size * size)/(float)Error_block.x), 1, 1); // размер сетки для запуска ядра на GPU
    

    double* itog; // указатель на выделенную память на GPU для хранения результата вычислений ядра reduceError
    cudaMalloc(&itog, sizeof(double) * Error_grid.x);

    cudaDeviceSynchronize(); // для синхронизации выполнения всех операций на устройстве CUDA

    while ((error > tol) && (num_iter < iter_max)){
        num_iter++;
        updateTemperature<<<Grid_Size, Block_size>>>(arr_pred_gp, arr_new_gp, size);
        if ((num_iter % 100 == 0) || (num_iter == 1)){
            error = 0.0;
            updateError<<<Grid_Size, Block_size>>>(arr_pred_gp, arr_new_gp, size, error, d_ptr); // ядро обновляет значения массивов arr_pred_gp и arr_new_gp
            
            cub::DeviceReduce::Max(tempStorage, tempStorageBytes, d_ptr, mas_error, (size) * (size)); //// нахождение максимума в разнице матрицы
           
            cudaMemcpy(&error, &mas_error[0], sizeof(double), cudaMemcpyDeviceToHost);

            d_ptr = arr_pred_gp;
            arr_pred_gp = arr_new_gp;
            arr_new_gp = d_ptr;

            printf("%d : %lf\n", num_iter, error);
            fflush(stdout); //  проверить, что все данные, которые были записаны в буфер вывода с помощью функции printf(), записались
        }
        else {
            
            d_ptr = arr_pred_gp;
            arr_pred_gp = arr_new_gp;
            arr_new_gp = d_ptr;
        }
    }
    printf("Финальные результаты: %d, %0.6lf\n", num_iter, error);
    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC; // переводит в секунды
    printf("%.25f время в секундах", d);
    return 0;
}
