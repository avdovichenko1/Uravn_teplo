#include <cstdlib>
#include <cstdio>
#include <malloc.h>
#include <time.h>

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
    shared_array[thread_id] = tol; // значение tol сохраняется в общей памяти
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

    int size_pot=32; // количество потоков
    dim3 Block_size(size_pot, size_pot, 1); //размер блока и определение количества потоков в каждом блоке, 1 блок - 1024 потока
    dim3 Grid_Size((size + 33)/size_pot, (size + 33)/size_pot, 1);

    double* d_A, *d_Anew;
    cudaMalloc((void**)&d_A, sizeof(double ) *(size+ 2)*(size+2));
    cudaMalloc((void**)&d_Anew, sizeof(double ) *(size+ 2)*(size+2));
    int iter_host = 0;
    double error_host = 1.0;
    double add_grad_host = 10.0 / (size + 2);

    int len_host = size + 2;
    for (int i = 0; i < size + 2; i++){
        arr_pred[i * len_host] = 10 + add_grad_host * i;
        arr_pred[i] = 10 + add_grad_host * i;
        arr_pred[len_host * (size + 1) + i] = 20 + add_grad_host * i;
        arr_pred[len_host * i + size + 1] = 20 + add_grad_host * i;

        arr_new[len_host * i] = arr_pred[i * len_host];
        arr_new[i] = arr_pred[i];
        arr_new[len_host * (size + 1) + i] = arr_pred[len_host * (size + 1) + i];
        arr_new[len_host * i + size + 1] = arr_pred[len_host * i + size + 1];
    }


    cudaMemcpy(d_A, arr_pred, sizeof(double) * (size + 2) * (size + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, arr_new, sizeof(double) * (size + 2) * (size + 2), cudaMemcpyHostToDevice);

    double* d_err_1d;
    cudaMalloc(&d_err_1d, sizeof(double) * (size * size));

    dim3 errBS(1024,1,1);
    dim3 errGS(ceil((size * size)/(float)errBS.x), 1, 1);
    double* dev_out;
    cudaMalloc(&dev_out, sizeof(double) * errGS.x);


    double* d_ptr;
    cudaMalloc((void **)(&d_ptr), sizeof(double));

    cudaDeviceSynchronize();
    while ((error_host > tol) && (iter_host < iter_max)){
        iter_host++;
        if ((iter_host % 150 == 0) || (iter_host == 1)){
            error_host = 0.0;
            updateError<<<Grid_Size, Block_size>>>(d_A, d_Anew, size, error_host, d_err_1d);
            reduceError<<<errGS, errBS, (errBS.x) * sizeof(double)>>>(d_err_1d, dev_out, size * size);
            reduceError<<<1, errBS, (errBS.x) * sizeof(double)>>>(dev_out, d_err_1d, errGS.x);
            cudaMemcpy(&error_host, &d_err_1d[0], sizeof(double), cudaMemcpyDeviceToHost);
        }
        else
            updateTemperature<<<Grid_Size, Block_size>>>(d_A, d_Anew, size);
        d_ptr = d_A;
        d_A = d_Anew;
        d_Anew = d_ptr;
        if ((iter_host % 150 == 0) || (iter_host == 1)){
            printf("%d : %lf\n", iter_host, error_host);
            fflush(stdout);
        }
    }
    printf("%d : %lf\n", iter_host, error_host);
    clock_t b=clock();
    double d=(double)(b-a)/CLOCKS_PER_SEC; // переводит в секунды
    printf("%.25f время в секундах", d);
    return 0;
}
