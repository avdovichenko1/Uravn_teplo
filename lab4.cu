#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define max(x, y) ((x) > (y) ? (x) : (y))


__global__ void updateTemperature(const double *arr_pred, double *arr_new, size_t N)
{
    int i = blockIdx.x + 1; // размер строки
    int j = threadIdx.x + 1; // столбца
    arr_new[i * N + j] = 0.25 * (arr_pred[i*N+j-1] + arr_pred[(i - 1) * N + j] +
                                 arr_pred[(i+1)*N+j] + arr_pred[i * N + j + 1]);
}


__global__ void update_matrix(const double* arr_pred, double* arr_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    arr_new[i] = arr_pred[i] - arr_new[i];
}

// Функция востановления границ матрицы
__global__ void restore(double* mas, int N){
    size_t i = threadIdx.x;
    double shag = 10.0 / (N-1);
    mas[i] = 10.0 + i * shag;
    mas[i * N] = 10.0 + i * shag;
    mas[N - 1 + i * N] = 20.0 + i * shag;
    mas[N * (N - 1) + i] = 20.0 + i * shag;
}

int main(int argc, char* argv[]) {
    clock_t a = clock();
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

    cudaSetDevice(1);

    int num_iter = 0;
    double error = 1.0;

    cudaStream_t stream; // указатель на объект потока CUDA
    cudaStreamCreate(&stream); // создание потока CUDA

    cudaGraph_t graph; //указатель на объект графа CUDA
    cudaGraphExec_t graph_exec; // указатель на объект выполнения графа CUDA

    double *arr_pred, *arr_new;
    cudaMalloc((void **)&arr_pred, sizeof(double) * size * size);
    cudaMalloc((void **)&arr_new, sizeof(double) * size * size);
    
    // Функция востановления границ матрицы
    {
        int i = threadIdx.x;
        double shag = 10.0 / (size-1);
        arr_pred[i] = 10.0 + i * shag;
        arr_pred[i * size] = 10.0 + i * shag;
        arr_pred[size - 1 + i * size] = 20.0 + i * shag;
        arr_pred[size * (size - 1) + i] = 20.0 + i * shag;
    }

    //restore<<<1, size>>>(arr_pred, size); //заполнение массива
    // копирование данных из хоста на устройство
    cudaMemcpy(arr_new, arr_pred, sizeof(double) * size * size, cudaMemcpyHostToDevice);

    // выделяем память на gpu. Хранение ошибки на device
    double *mas_error = 0;
    cudaMalloc((void **)&mas_error, sizeof(double)); //выделение памяти для GPU

    size_t tempStorageBytes = 0;
    double *tempStorage = NULL; // временного хранения буфера для операции редукции на GPU

    // получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, arr_new, mas_error, size * size, stream);

    cudaMalloc(&tempStorage, tempStorageBytes); //выделение памяти для буфера

    bool graphCreated = false;

    while ((iter_max > num_iter) && (error > tol)) {
        if(!graphCreated){
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            for (size_t i = 0; i < 100; i += 2) {
                updateTemperature<<<size - 2, size - 2, 0, stream>>>(arr_pred, arr_new, size);
                updateTemperature<<<size - 2, size - 2, 0, stream>>>(arr_new, arr_pred, size);
            }
            update_matrix<<<size, size, 0, stream>>>(arr_pred, arr_new);

            cub::DeviceReduce::Max(tempStorage, tempStorageBytes, arr_new, mas_error, size * size, stream);
            restore<<<1, size, 0, stream>>>(arr_new, size);

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
            graphCreated=true;

        }
        else{
            cudaGraphLaunch(graph_exec, stream);
            cudaMemcpyAsync(&error, mas_error, sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            num_iter+=100;
            graphCreated=false;
        }

    }

    printf("Финальные результаты: %d, %0.6lf\n", num_iter, error);

    // удаление потока и графа
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);

    cudaFree(arr_pred);
    cudaFree(arr_new);

    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC; // переводит в секунды
    printf("%.25f время в секундах", d);

    return 0;
}
