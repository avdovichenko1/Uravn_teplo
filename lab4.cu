#include <cstdlib>
#include <cstdio>
#include <malloc.h>
#include <time.h>

__global__ void heat(const double* arr_pred, double* arr_new, int N){
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (a > 0 && a < N + 1 && b > 0 && b < N + 1)
        arr_new[a * (N + 2) + b] = 0.25 * (arr_pred[(a + 1) * (N + 2) + b] + arr_pred[(a - 1) * (N + 2) + b] + arr_pred[a * (N + 2) + b - 1] + arr_pred[a * (N + 2) + b + 1]);
}

#define max(x, y) ((x) > (y) ? (x) : (y))

__global__ void heatError(const double* arr_pred, double* arr_new, int N, double tol, double* tol1){
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (a > 0 && a < N + 1 && b > 0 && b < N + 1){
        arr_new[a * (N + 2) + b] = 0.25 * (arr_pred[(a + 1) * (N + 2) + b] + arr_pred[(a - 1) * (N + 2) + b] + arr_pred[a * (N + 2) + b - 1] + arr_pred[a * (N + 2) + b + 1]);
        int idx_1d = (b * a) - 1;
        tol1[idx_1d] = max(arr_new[a * (N + 2) + b] - arr_pred[a * (N + 2) + b], tol);
    }
}
__global__ void errorReduce(double* er_1d, double* er_blocks, int size){
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
    int max_num_iter = atoi(argv[1]); // количество итераций
    double max_toch = atof(argv[2]); // точность
    int raz = atoi(argv[3]); // размер сетки
    clock_t a=clock();

    double *arr_pred = (double *) malloc((raz + 2) * (raz + 2) * sizeof(double));
    double *arr_new = (double *) malloc((raz + 2) * (raz + 2) * sizeof(double));
    dim3
            BS(32, 32, 1);
    dim3
            GS((raz + 2 + 31) / 32, (raz + 2 + 31) / 32, 1);


    double *d_A, *d_Anew;
    cudaMalloc((void **) &d_A, sizeof(double) * (raz + 2) * (raz + 2));
    cudaMalloc((void **) &d_Anew, sizeof(double) * (raz + 2) * (raz + 2));
    int iter_host = 0;
    double error_host = 1.0;
    double add_grad_host = 10.0 / (raz + 2);

    int len_host = raz + 2;
    for (int i = 0; i < raz + 2; i++) {
        arr_pred[i * len_host] = 10 + add_grad_host * i;
        arr_pred[i] = 10 + add_grad_host * i;
        arr_pred[len_host * (raz + 1) + i] = 20 + add_grad_host * i;
        arr_pred[len_host * i + raz + 1] = 20 + add_grad_host * i;

        arr_new[len_host * i] = arr_pred[i * len_host];
        arr_new[i] = arr_pred[i];
        arr_new[len_host * (raz + 1) + i] = arr_pred[len_host * (raz + 1) + i];
        arr_new[len_host * i + raz + 1] = arr_pred[len_host * i + raz + 1];
    }


    cudaMemcpy(d_A, arr_pred, sizeof(double) * (raz + 2) * (raz + 2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, arr_new, sizeof(double) * (raz + 2) * (raz + 2), cudaMemcpyHostToDevice);

    double *d_err_1d;
    cudaMalloc(&d_err_1d, sizeof(double) * (raz * raz));

    dim3
            errBS(1024, 1, 1);
    dim3
            errGS(ceil((raz * raz) / (float) errBS.x), 1, 1);
    double *dev_out;
    cudaMalloc(&dev_out, sizeof(double) * errGS.x);


    double *d_ptr;
    cudaMalloc((void **) (&d_ptr), sizeof(double));

    cudaDeviceSynchronize();
    while ((error_host > max_toch) && (iter_host < max_num_iter)) {
        iter_host++;
        if ((iter_host % 150 == 0) || (iter_host == 1)) {
            error_host = 0.0;
            heatError<<<GS, BS>>>(d_A, d_Anew, raz, error_host, d_err_1d);
            errorReduce<<<errGS, errBS, (errBS.x) * sizeof(double)>>>(d_err_1d, dev_out, raz * raz);
            errorReduce<<<1, errBS, (errBS.x) * sizeof(double)>>>(dev_out, d_err_1d, errGS.x);
            cudaMemcpy(&error_host, &d_err_1d[0], sizeof(double), cudaMemcpyDeviceToHost);
        } else
            heat<<<GS, BS>>>(d_A, d_Anew, raz);
        d_ptr = d_A;
        d_A = d_Anew;
        d_Anew = d_ptr;
        if ((iter_host % 150 == 0) || (iter_host == 1)) {
            printf("%d : %lf\n", iter_host, error_host);
            fflush(stdout);
        }
    }
    printf("%d : %lf\n", iter_host, error_host);
    return 0;
}