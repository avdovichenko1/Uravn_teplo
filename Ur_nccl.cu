#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "nccl.h"
#include "mpi.h"

#define CORNER1 10
#define CORNER2 20
#define CORNER3 30
#define CORNER4 20


__global__ void updateTemperature(double* arr_pred, double* arr_new, int N, size_t sizePerGpu){
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(!(j == 0 || i == 0 || j == N - 1 || i == sizePerGpu - 1))
        arr_new[i * N + j] = 0.25 * (arr_pred[i * N + j - 1] + arr_pred[(i - 1) * N + j] + arr_pred[(i + 1) * N + j] + arr_pred[i * N + j + 1]);
}

// Функция, подсчитывающая разницу матриц
__global__ void update_matrix(double* arr_pred, double* arr_new, double* out, int N){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = std::abs(arr_new[i] - arr_pred[i]);
}

int main(int argc, char** argv) {
    int processRank; // номер текущего процесса
    int groupSize; // общее количество процессов
    MPI_Init(&argc, &argv); // инициализация MPI-окружения

    MPI_Comm_rank(MPI_COMM_WORLD, &processRank); // определение номера текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &groupSize); // определение общего количества процессов

    cudaSetDevice(processRank); // каждый процесс назначает себе соответствующее CUDA-устройство

    const double tolerance = std::pow(10, -std::stoi(argv[1])); // заданная точность
    const int gridSize = std::stoi(argv[2]); // размер сетки
    const int maxIterations = std::stoi(argv[3]); // максимальное количество итераций

    const size_t totalSize = gridSize * gridSize; // общий размер матрицы

    if (processRank == 0) {
        std::cout << "Параметры: " << std::endl <<
                  "Точность: " << tolerance << std::endl <<
                  "Максимальное число итераций: " << maxIterations << std::endl <<
                  "Размер сетки: " << gridSize << std::endl;
    }

    if (processRank != 0) {
        cudaDeviceEnablePeerAccess(processRank - 1, 0); // позволяют текущему процессу установить соединение и получить доступ к памяти на этом соседнем устройствах CUDA
    }

    if (processRank != groupSize - 1) {
        cudaDeviceEnablePeerAccess(processRank + 1, 0);
    }

    ncclUniqueId id;
    ncclComm_t comm;
    if (processRank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, groupSize, id, processRank);

// Разделение границ между устройствами
    size_t areaSizePerProcess = gridSize / groupSize;
    size_t startYIndex = areaSizePerProcess * processRank;

// Выделение памяти на хосте
    double *matrixA, *matrixB;
    cudaMallocHost(&matrixA, sizeof(double) * totalSize);
    cudaMallocHost(&matrixB, sizeof(double) * totalSize);

    std::memset(matrixA, 0, gridSize * gridSize * sizeof(double));

// Заполнение граничных условий
    matrixA[0] = CORNER1;
    matrixA[gridSize - 1] = CORNER2;
    matrixA[gridSize * gridSize - 1] = CORNER3;
    matrixA[gridSize * (gridSize - 1)] = CORNER4;

    const double step = 1.0 * (CORNER2 - CORNER1) / (gridSize - 1);
    for (int i = 1; i < gridSize - 1; i++) {
        matrixA[i] = CORNER1 + i * step;
        matrixA[i * gridSize] = CORNER1 + i * step;
        matrixA[gridSize - 1 + i * gridSize] = CORNER2 + i * step;
        matrixA[gridSize * (gridSize - 1) + i] = CORNER4 + i * step;
    }

    std::memcpy(matrixB, matrixA, totalSize * sizeof(double));

    double *deviceMatrixA, *deviceMatrixB, *deviceError, *errorMatrix, *tempStorage = NULL;

// Вычисление необходимого объема памяти для каждого процесса
    if (processRank != 0 && processRank != groupSize - 1) {
        areaSizePerProcess += 2;
    } else {
        areaSizePerProcess += 1;
    }

    size_t allocatedMemorySize = gridSize * areaSizePerProcess;

// Выделение памяти на устройстве
    cudaMalloc((void **) &deviceMatrixA, allocatedMemorySize * sizeof(double));
    cudaMalloc((void **) &deviceMatrixB, allocatedMemorySize * sizeof(double));
    cudaMalloc((void **) &errorMatrix, allocatedMemorySize * sizeof(double));
    cudaMalloc((void **) &deviceError, sizeof(double));

// Копирование заполненной матрицы в выделенную память, начиная со 2 строки
    size_t offset = (processRank != 0) ? gridSize : 0;
    cudaMemcpy(deviceMatrixA, matrixA + (startYIndex * gridSize) - offset, sizeof(double) * allocatedMemorySize,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB + (startYIndex * gridSize) - offset, sizeof(double) * allocatedMemorySize,
               cudaMemcpyHostToDevice);

// Определение размера временного буфера для редукции и выделение памяти для этого буфера
    size_t tempStorageSize = 0;
    cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, gridSize * areaSizePerProcess);
    cudaMalloc((void **) &tempStorage, tempStorageSize);

    int iterations = 0;
    double *error;
    cudaMallocHost(&error, sizeof(double));
    *error = 1.0;

    unsigned int threadsX = (gridSize < 1024) ? gridSize : 1024;
    unsigned int blocksY = areaSizePerProcess;
    unsigned int blocksX = gridSize / threadsX;

    dim3 blockDim(threadsX, 1);
    dim3 gridDim(blocksX, blocksY);

    cudaStream_t stream, memoryStream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&memoryStream);

// Основной алгоритм
    clock_t begin = clock();
    while (iterations < maxIterations && (*error) > tolerance) {
        iterations++;

        // Расчет матрицы
        updateTemperature<<<gridDim, blockDim, 0, stream>>>(deviceMatrixA, deviceMatrixB, gridSize, areaSizePerProcess);

        // Вычисление ошибки каждые 100 итераций
        if (iterations % 100 == 0) {
            updateMatrix<<<blocksX * blocksY, threadsX, 0, stream>>>(deviceMatrixA, deviceMatrixB, errorMatrix,
                                                                     gridSize);
            cub::DeviceReduce::Max(tempStorage, tempStorageSize, errorMatrix, deviceError, allocatedMemorySize, stream);

            ncclAllReduce((void *) deviceError, (void *) deviceError, 1, ncclDouble, ncclMax, comm, stream);

            cudaMemcpyAsync(error, deviceError, sizeof(double), cudaMemcpyDeviceToHost, stream);
        }

        cudaStreamSynchronize(stream);

        // Обмен "граничными" условиями каждой области
        // Обмен верхней границей
        ncclGroupStart();
        if (processRank != 0) {
            ncclSend(deviceMatrixB + gridSize + 1, gridSize - 2, ncclDouble, processRank - 1, comm, stream);
            ncclRecv(deviceMatrixB + 1, gridSize - 2, ncclDouble, processRank - 1, comm, stream);
        }
        // Обмен нижней границей
        if (processRank != groupSize - 1) {
            ncclSend(deviceMatrixB + (areaSizePerProcess - 2) * gridSize + 1,
                     gridSize - 2, ncclDouble, processRank + 1, comm, stream);
            ncclRecv(deviceMatrixB + (areaSizePerProcess - 1) * gridSize + 1,
                     gridSize - 2, ncclDouble, processRank + 1, comm, stream);
        }
        ncclGroupEnd();

        // Обмен указателями
        std::swap(deviceMatrixA, deviceMatrixB);
    }

    clock_t end = clock();
    if (processRank == 0) {
        std::cout << "Time: " << 1.0 * (end - begin) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Iterations: " << iterations << " Error: " << *error << std::endl;
    }

// Освобождение памяти
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(errorMatrix);
    cudaFree(tempStorage);
    cudaFree(matrixA);
    cudaFree(matrixB);

    MPI_Finalize();

    return 0;
}
