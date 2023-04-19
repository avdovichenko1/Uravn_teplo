#include <cstdlib>
#include <cstdio>
#include <malloc.h>
#include <time.h>

#define max(x, y) ((x) > (y) ? (x) : (y))


__global__ void updateTemperature(const double* arr_pred, double* arr_new, int N){
    //Индекс i вычисляется как произведение номера блока по вертикальной оси (blockIdx.y) на размер блока по вертикальной оси
    // (blockDim.y),плюс номер потока внутри блока по вертикальной оси (threadIdx.y), что позволяет потокам различных блоков
    // и потокам внутри одного блока работать с различными строками массива данных, j - аналогично.
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

//проверяется, что индексы i и j находятся внутри диапазона от 1 до N + 1, чтобы исключить обработку граничных элементов массива.
    if (i > 0 && i < N + 1)
        if (j > 0 && j < N + 1)
            //новое значение элемента массива arr_new[i * (N + 2) + j] вычисляется на основе предыдущего состояния массива
            // arr_pred,используя формулу теплопроводности
            arr_new[i * (N + 2) + j] = 0.25 * (arr_pred[(i + 1) * (N + 2) + j] + arr_pred[(i - 1) * (N + 2) + j] +
                                               arr_pred[i * (N + 2) + j - 1] + arr_pred[i * (N + 2) + j + 1]);
}

__global__ void updateError(const double* arr_pred, double* arr_new, int N, double tol, double* tol1){

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < N + 1)
        if (j > 0 && j < N + 1) {
            arr_new[i * (N + 2) + j] = 0.25 * (arr_pred[(i + 1) * (N + 2) + j] + arr_pred[(i - 1) * (N + 2) + j] + arr_pred[i * (N + 2) + j - 1] + arr_pred[i * (N + 2) + j + 1]);
            //Вычисление значения погрешности между новым значением элемента и соответствующим предыдущим значением элемента
            tol1[j * i - 1] = max(arr_new[i * (N + 2) + j] - arr_pred[i * (N + 2) + j], tol);
        };
}

__global__ void reduceError(double* er_1d, double* er_blocks, int size){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int gsz = blockDim.x * gridDim.x;
    double error = er_1d[0];
    for (int i  = gid; i < size; i+= gsz)
        error = max(error, er_1d[i]);
    extern __shared__ double shArr[];
    shArr[tid] = error;
    __syncthreads();
    for (int sz = blockDim.x / 2; sz > 0; sz /=2){
        if (tid < sz)
            shArr[tid] = max(shArr[tid + sz], shArr[tid]);
        __syncthreads();
    }
    if (tid == 0)
        er_blocks[blockIdx.x] = shArr[0];
}


int main(int argc, char* argv[]) {
    clock_t a=clock();
    int size;
    double tol;
    int iter_max;
    if (argc < 4){
        printf("Not enough args");
        exit(1);
    }
    else
    {
        tol = strtod(argv[1], NULL);
        if (tol == 0){
            printf("incorrect first param");
            exit(1);
        }
        size = atoi(argv[2]);
        if (size == 0){
            printf("incorrect second param");
            exit(1);
        }
        iter_max = atoi(argv[3]);
        if (iter_max == 0){
            printf("incorrect third param");
            exit(1);
        }
    }
    double *A = (double*)malloc((size + 2)*(size + 2) * sizeof(double));
    double *Anew = (double*)malloc((size + 2)*(size + 2) * sizeof(double));
    dim3 BS(32, 32, 1);
    dim3 GS((size + 2 + 31)/32, (size + 2 + 31)/32, 1);


    double* d_A, *d_Anew;
    cudaMalloc((void**)&d_A, sizeof(double ) *(size+ 2)*(size+2));
    cudaMalloc((void**)&d_Anew, sizeof(double ) *(size+ 2)*(size+2));
    int iter_host = 0;
    double error_host = 1.0;
    double add_grad_host = 10.0 / (size + 2);

    int len_host = size + 2;
    for (int i = 0; i < size + 2; i++)
    {
        A[i * len_host] = 10 + add_grad_host * i;
        A[i] = 10 + add_grad_host * i;
        A[len_host * (size + 1) + i] = 20 + add_grad_host * i;
        A[len_host * i + size + 1] = 20 + add_grad_host * i;

        Anew[len_host * i] = A[i * len_host];
        Anew[i] = A[i];
        Anew[len_host * (size + 1) + i] = A[len_host * (size + 1) + i];
        Anew[len_host * i + size + 1] = A[len_host * i + size + 1];
    }


    cudaMemcpy( d_A, A, sizeof(double)*(size + 2) * (size + 2), cudaMemcpyHostToDevice);
    cudaMemcpy( d_Anew, Anew, sizeof(double)*(size + 2) * (size + 2), cudaMemcpyHostToDevice);

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
            updateError<<<GS, BS>>>(d_A, d_Anew, size, error_host, d_err_1d);
            reduceError<<<errGS, errBS, (errBS.x) * sizeof(double)>>>(d_err_1d, dev_out, size * size);
            reduceError<<<1, errBS, (errBS.x) * sizeof(double)>>>(dev_out, d_err_1d, errGS.x);
            cudaMemcpy(&error_host, &d_err_1d[0], sizeof(double), cudaMemcpyDeviceToHost);
        }
        else
            updateTemperature<<<GS, BS>>>(d_A, d_Anew, size);
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
