#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "nccl.h"
#include "mpi.h"

__global__ void updateTemperature(double* arr_pred, double* arr_new, int N, size_t sizePerGpu){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N - 1 && j < N - 1 && i > 0 && j > 0) { {
            arr_new[i * N + j] = 0.25 * (arr_pred[i * N + j - 1] + arr_pred[(i - 1) * N + j] +
                                         arr_pred[(i + 1) * N + j] + arr_pred[i * N + j + 1]);
        }
    }

// Функция, подсчитывающая разницу матриц
    __global__ void update_matrix(double* arr_pred, double* arr_new, double* out, int N){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        out[i] = fabs(arr_new[i] - arr_pred[i]);
    }

int main(int argc, char** argv) {
    clock_t a = clock();
    int processRank; // номер текущего процесса
    int groupSize; // общее количество процессов
    MPI_Init(&argc, &argv); // инициализация MPI-окружения
    
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank); // определение номера текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &groupSize); // определение общего количества процессов

    cudaSetDevice(processRank); // каждый процесс назначает себе соответствующее CUDA-устройство
    
    if (argc < 4){
        printf("Неправильное количество аргументов");
        exit(1);
    }
    tolerance = strtod(argv[1], NULL);
    if (tolerance <= 0){
        printf("Ограничение точности должно превышать 0");
        exit(1);
    }
    gridSize = atoi(argv[2]);
    if (gridSize <= 0){
        printf("Размер матриццы должен быть больше 0");
        exit(1);
    }
    maxIterations = atoi(argv[3]);
    if (maxIterations <= 0){
        printf("Максимальное количество итераци должно быть больше 0");
        exit(1);
    }

    if (processRank != 0)
        cudaDeviceEnablePeerAccess(processRank - 1, 0); //вызывается функция cudaDeviceEnablePeerAccess для разрешения доступа между текущим процессом и предыдущим

    if (processRank != groupSize - 1)
        cudaDeviceEnablePeerAccess(processRank + 1, 0); //доступ между устройствами на разных процессах, чтобы они могли обмениваться данными при необходимости

    ncclUniqueId id;
    ncclComm_t comm;

    if (processRank == 0) {
        ncclGetUniqueId(&id); //уникальный идентификатор id генерируется только для процесса с рангом 0
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD); // уникальный идентификатор id передается остальным процессам
    ncclCommInitRank(&comm, groupSize, id, processRank); //создается коммуникатор comm для группы процессов

    size_t areaSizePerProcess = gridSize / groupSize; // получение размера области данных на каждом процессе.

    double* arr_pred, * arr_new;
    cudaMallocHost(&arr_pred, sizeof(double) * gridSize * gridSize);
    cudaMallocHost(&arr_new, sizeof(double) * gridSize * gridSize);
    memset(arr_pred, 0, gridSize * gridSize * sizeof(double)); // заполняет указанную область памяти нулями

    double shag = 10.0 / (gridSize - 1);

    for (int i = 0; i < gridSize; i++) {
        arr_pred[i] = 10.0 + i * shag;
        arr_pred[i * gridSize] = 10.0 + i * shag;
        arr_pred[i * gridSize + gridSize - 1] = 20.0 + i * shag;
        arr_pred[gridSize * (gridSize - 1) + i] = 20.0 + i * shag;
    }

    memcpy(arr_new, arr_pred, gridSize * gridSize * sizeof(double));

    double* device_arr_pred, * device_arr_new, * deviceError, * errorMatrix, * tempStorage = NULL;

    //Определяется размер области данных, которая будет обрабатываться каждым процессом в группе процессов.
    if (processRank == 0 || processRank == groupSize - 1) {
        areaSizePerProcess += 1;
    } else {
        areaSizePerProcess += 2;
    }

    size_t allocatedMemorySize = gridSize * areaSizePerProcess; //вычисляет общий размер выделенной памяти для массивов данных

    cudaMalloc((void**)&device_arr_pred, allocatedMemorySize * sizeof(double));
    cudaMalloc((void**)&device_arr_new, allocatedMemorySize * sizeof(double));
    cudaMalloc((void**)&errorMatrix, allocatedMemorySize * sizeof(double));
    cudaMalloc((void**)&deviceError, sizeof(double));

    size_t tempStorageSize = 0;
    cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, gridSize * areaSizePerProcess);
    cudaMalloc((void**)&tempStorage, tempStorageSize);

    int num_iterations = 0;
    double* error;
    cudaMallocHost(&error, sizeof(double));
    *error = 1.0;

    unsigned int threadsX = (gridSize < 1024) ? gridSize : 1024;
    unsigned int blocksY = areaSizePerProcess;
    unsigned int blocksX = gridSize / threadsX;

    dim3 blockDim(threadsX, 1);
    dim3 gridDim(blocksX, blocksY);

    cudaStream_t stream; //используется для запуска ядер CUDA и асинхронной копии данных между хостом и устройством
    cudaStreamCreate(&stream);

    cudaStream_t memoryStream; //используется для выполнения операции синхронизации
    cudaStreamCreate(&memoryStream);

    while (num_iterations < maxIterations && (*error) > tolerance) {
        num_iterations++;

        updateTemperature<<<gridDim, blockDim, 0, stream>>>(device_arr_pred, device_arr_new, gridSize, areaSizePerProcess);

        if (num_iterations % 100 == 0) {
            update_matrix<<<blocksX * blocksY, threadsX, 0, stream>>>(device_arr_pred, device_arr_new, errorMatrix, gridSize);
            cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, allocatedMemorySize, stream);

            ncclAllReduce((void*)deviceError, (void*)deviceError, 1, ncclDouble, ncclMax, comm, stream); //выполняется операция максимума среди всех процессов
// 1 - у нас 1 элемент deviceerror

            cudaMemcpyAsync(error, deviceError, sizeof(double), cudaMemcpyDeviceToHost, stream);
        }

        cudaStreamSynchronize(stream);

        ncclGroupStart(); // начинает группу операций NCCL
        if (processRank != 0) {
            //процесс отправляет данные смещением gridSize + 1 из device_arr_new размером gridSize - 2 элементов типа ncclDouble процессу с рангом processRank - 1
            ncclSend(device_arr_new + gridSize + 1, gridSize - 2, ncclDouble, processRank - 1, comm, stream);
            //принимает данные размером gridSize - 2 элементов типа ncclDouble от процесса с рангом processRank - 1
            ncclRecv(device_arr_new + 1, gridSize - 2, ncclDouble, processRank - 1, comm, stream);
        }
        if (processRank != groupSize - 1) {
            ncclSend(device_arr_new + (areaSizePerProcess - 2) * gridSize + 1, gridSize - 2, ncclDouble, processRank + 1, comm, stream);
            ncclRecv(device_arr_new + (areaSizePerProcess - 1) * gridSize + 1, gridSize - 2, ncclDouble, processRank + 1, comm, stream);
        }
        ncclGroupEnd();

        double* temp = device_arr_pred;
        device_arr_pred = device_arr_new;
        device_arr_new = temp;
    }

    clock_t b = clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC; // переводит в секунды
    
    if (processRank == 0) {
        printf("Финальные результаты: %d, %0.6lf\n", num_iterations, *error);
        printf("%.25f время в секундах", d);
    }

    cudaFree(device_arr_pred);
    cudaFree(device_arr_new);
    cudaFree(errorMatrix);
    cudaFree(tempStorage);
    cudaFree(arr_pred);
    cudaFree(arr_new);

    MPI_Finalize();

    return 0;
}
