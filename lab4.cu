#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define max(x, y) ((x) > (y) ? (x) : (y))


__global__ void restore(double* mas, int N){
    size_t i = threadIdx.x;
    double shag = 10.0 / (N-1);
    if (i < N - 1 && i > 0){
        mas[i] = 10.0 + i * shag;
        mas[i * N] = 10.0 + i * shag;
        mas[N - 1 + i * N] = 20.0 + i * shag;
        mas[N * (N - 1) + i] = 20.0 + i * shag;
    }
}


__global__ void updateTemperature(const double *arr_pred, double *arr_new, size_t N){
    int i = blockIdx.x;
    int j = threadIdx.x; 
    if (i < N - 1 && j < N - 1 && i > 0 && j < 0) {
        arr_new[i * N + j] = 0.25 * (arr_pred[i*N+j-1] + arr_pred[(i - 1) * N + j] +
                                 arr_pred[(i+1)*N+j] + arr_pred[i * N + j + 1]);
    }
}


__global__ void update_matrix(const double* arr_pred, double* arr_new){
    int i = blockIdx.x * blockDim.x + threadIdx.x; //вычисления линейного индекса элемента внутри сетки CUDA
    if (i < N - 1 && i > 0){
        arr_new[i] = arr_pred[i] - arr_new[i];
    }
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

    cudaSetDevice(0);

    int num_iter = 0;
    double error = 1.0;

    cudaStream_t stream; // указатель на объект потока CUDA
    cudaStreamCreate(&stream); // создание потока CUDA

    cudaGraph_t graph; //указатель на объект графа CUDA
    cudaGraphExec_t graph_exec; // указатель на объект выполнения графа CUDA

    double *arr_pred, *arr_new;
    cudaMalloc((void **)&arr_pred, sizeof(double) * size * size);
    cudaMalloc((void **)&arr_new, sizeof(double) * size * size);
    
    // Выделение памяти на хосте
    double* host_arr_pred = (double*)malloc(sizeof(double) * size * size);

    // Заполнение границ массива
    double shag = 10.0 / (size - 1);
    for (size_t i = 0; i < size; i++) {
        host_arr_pred[i] = 10.0 + i * shag;
        host_arr_pred[i * size] = 10.0 + i * shag;
        host_arr_pred[size - 1 + i * size] = 20.0 + i * shag;
        host_arr_pred[size * (size - 1) + i] = 20.0 + i * shag;
    }

    // Копирование данных из хоста в устройство
    cudaMemcpy(arr_pred, host_arr_pred, sizeof(double) * size * size, cudaMemcpyHostToDevice);
    free(host_arr_pred); // Освобождение памяти на хосте
    
    // копирование данных из хоста на устройство
    cudaMemcpy(arr_new, arr_pred, sizeof(double) * size * size, cudaMemcpyHostToDevice);

    // выделяем память на gpu. Хранение ошибки на device
    double *mas_error = 0;
    cudaMalloc((void **)&mas_error, sizeof(double)); //выделение памяти для GPU

    size_t tempStorageBytes = 0;
    double *tempStorage = NULL; // временного хранения буфера для операции редукции на GPU
    
   dim3 thread = size < 1024 ? size : 1024;
    dim3 block = size / (size < 1024 ? size : 1024);

    // получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, arr_new, mas_error, size * size, stream);

    cudaMalloc(&tempStorage, tempStorageBytes); //выделение памяти для буфера
    //
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal); //записывает операции, выполняемые в потоке

    for (size_t i = 0; i < 100; i += 2) {
        updateTemperature<<<block, thread, 0, stream>>>(arr_pred, arr_new, size); // количество потоков в блоке, количество блоков, разделяемая память
        updateTemperature<<<block, thread, 0, stream>>>(arr_new, arr_pred, size);
    }
            
    update_matrix<<<block, thread, 0, stream>>>(arr_pred, arr_new);

    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, arr_new, mas_error, size * size, stream);
    restore<<<1, size, 0, stream>>>(arr_new, size);
        

    cudaStreamEndCapture(stream, &graph); //завершение захвата операций    
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0); // создание граф выполнения
       

    while ((iter_max > num_iter) && (error > tol)) {
        cudaGraphLaunch(graph_exec, stream); // его запуск
        cudaStreamSynchronize(stream);
        num_iter+=100;
        
        printf("%d : %lf\n", num_iter, error);
        fflush(stdout); //  проверить, что все данные, которые были записаны в буфер вывода с помощью функции printf(), записались
    }


    printf("Финальные результаты: %d, %0.6lf\n", num_iter, error);
   
   // Копирование данных из устройства на хост
   double* host_arr_pred_1 = (double*)malloc(sizeof(double) * size * size);
   cudaMemcpy(host_arr_pred_1, arr_pred, sizeof(double) * size * size, cudaMemcpyDeviceToHost);

   // Вывод матрицы на экран
   printf("Матрица arr_pred после выполнения операций:\n");
   for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
         printf("%0.2lf ", host_arr_pred_1[i * size + j]);
      }
      printf("\n");
   }

   free(host_arr_pred_1); // Освобождение памяти на хосте

    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);

    cudaFree(arr_pred);
    cudaFree(arr_new);

    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC; // переводит в секунды
    printf("%.25f время в секундах", d);

    return 0;
}
