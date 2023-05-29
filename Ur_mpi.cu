#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"

// функция изменения матрицы уравнения теплопроводности
__global__ void calculate(double *CudaArr, double *CudaNewArr, size_t MatrixX, size_t MatrixY)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x; //вычисления линейного индекса элемента внутри сетки 
    size_t j =  blockDim.y * blockIdx.y + threadIdx.y; 
    int index = i * MatrixX + j;
    if ((i < MatrixX - 1 && j < MatrixY - 1 && i > 0 && j > 0)) 
        CudaNewArr[index] = 0.25 * (CudaArr[(i - 1) * MatrixX + j] + CudaArr[(i + 1) * MatrixX + j] + CudaArr[index - 1] + CudaArr[index + 1]);
}


// функция разницы матриц
__global__ void subtraction(double* CudaArr, double* CudaNewArr, double* CudaArrErr, size_t Matrix)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  
    size_t j =  blockDim.y * blockIdx.y + threadIdx.y;
    int idx = i * Matrix + j; 
    if ((i < Matrix && j < Matrix && i > 0 && j > 0))
	    CudaArrErr[idx] = CudaArr[idx] - CudaNewArr[idx];
}

// функция востановления границ матрицы
__global__ void restore(double* arr, int size){
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i >= size-1)
        	return;
	
	arr[i] = 10.0 + i * 10.0 / (size - 1);
	arr[i * size] = 10.0 + i * 10.0 / (size - 1);
	arr[size - 1 + i * size] = 20.0 + i * 10.0 / (size - 1);
	arr[size * (size - 1) + i] = 20.0 + i * 10.0 / (size - 1);
}

int find_threads(int size){
	if (size%32==0)
		return size/1024;

	return int(size/1024)+1;

}

int main(int argc, char* argv[]) {
    
    double time_spent = 0.0;
    clock_t begin = clock(); 

    // Convert command line arguments to integers
    int Matrix = atoi(argv[1]);
    double accuracy = atof(argv[2]);
    int iterations = atoi(argv[3]);

    int rank, size;
    /* Initialize the MPI library */
    MPI_Init(&argc,&argv);
    /* Determine the calling process rank and total number of ranks */
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    /* Call MPI routines like MPI_Send, MPI_Recv, ... */
    cudaSetDevice(rank);       

    if (rank != 0)
        cudaDeviceEnablePeerAccess(rank - 1, 0);
    if (rank != (size-1))
        cudaDeviceEnablePeerAccess(rank + 1, 0);
	
    size_t size_y = Matrix / size + 1;
    if (rank != size - 1 && rank != 0) 
	    size_y += 1;
	
    dim3 t(32,32); //определяю количество нитей в каждом блоке
    dim3 b(find_threads(Matrix), find_threads(Matrix)); // количество блоков
	
    // выделяем память на gpu через cuda для 3 сеток
    double *A, *CudaArr, *CudaNewArr, *CudaArrErr;
    cudaMalloc((void **)&CudaArr, sizeof(double) * Matrix * size_y);
    cudaMalloc((void **)&CudaNewArr, sizeof(double) * Matrix * size_y);
    cudaMalloc((void **)&CudaArrErr, sizeof(double) * Matrix * size_y);

    cudaMallocHost(&A, sizeof(double) * Matrix * Matrix);
    restore<<<b, t>>>(A, Matrix);
	
    size_t offset = (rank != 0) ? Matrix : 0;
    cudaMemcpy(CudaArr, A + (Matrix * Matrix * rank / size) - offset, sizeof(double) * Matrix * size_y, cudaMemcpyHostToDevice);
    cudaMemcpy(CudaNewArr, A + (Matrix * Matrix * rank / size) - offset, sizeof(double) * Matrix * size_y, cudaMemcpyHostToDevice);

    // выделяем память на gpu. Хранение ошибки на device
    double *max_err = 0;
    cudaMalloc((void **)&max_err, sizeof(double));

    size_t tempStorageBytes = 0;
    double *tempStorage = NULL;

    // получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaNewArr, max_err, Matrix * size_y);

    // выделяем память для буфера
    cudaMalloc(&tempStorage, tempStorageBytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);	

    // Main loop
    double err = 1;
    int iter = 0;

    while (err > accuracy && iter < iterations) 
    {
	iter++;
	calculate <<<b, t, 0, stream>>> (CudaArr, CudaNewArr, Matrix, size_y);
	// Расчитываем ошибку каждую сотую итерацию
	if (iter % 100 == 0) {
		subtraction<<<b, t, 0, stream>>>(CudaArr, CudaNewArr, CudaArrErr, Matrix);
		cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaArrErr, max_err, Matrix * size_y);
		cudaMemcpy(&err, max_err, sizeof(double), cudaMemcpyDeviceToHost);

		MPI_Allreduce((void*)&max_err,(void*)&max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		
		cudaMemcpyAsync(&err, max_err, sizeof(double), cudaMemcpyDeviceToHost, stream); // запись ошибки в переменную на host
            	// Находим максимальную ошибку среди всех и передаём её всем процессам
	}

	if (rank != 0){ // Обмен верхней границей
            	MPI_Sendrecv(CudaNewArr + Matrix + 1, Matrix - 2, MPI_DOUBLE, rank - 1, 0, CudaNewArr + 1, 
			     Matrix - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}	
        if (rank != size - 1) { // Обмен нижней границей
            	MPI_Sendrecv(CudaNewArr + (size_y - 2) * Matrix + 1, Matrix - 2, MPI_DOUBLE, rank + 1, 0, 
			     CudaNewArr + (size_y - 1) * Matrix + 1, Matrix - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}    
	double* c = CudaNewArr; 
        CudaNewArr = CudaArr;
        CudaArr = c;
    }

    printf("Final result: %d, %0.6lf\n", iter, err);


    cudaFree(CudaArr);
    cudaFree(CudaNewArr);
    cudaFree(CudaArrErr);
    cudaFree(A);

    MPI_Finalize();
	
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f\n", time_spent);

    return 0;
}
