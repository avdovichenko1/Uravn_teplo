#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

cublasHandle_t handle; // для работы с библиотекой CUBLAS

__global__ void sigmoid(float* x) {
    int idx = threadIdx.x;
    x[idx] = exp(x[idx]) / (1 + exp(x[idx]));
}

class Linear {
    float* weights;
    float* biases;
    int in_features;
    int out_features;
public:
    Linear() {
        weights = NULL;
        biases = NULL;
        in_features = 0; // размерность входа
        out_features = 0; // размерность выхода
    };
    Linear(int in, int out) {
        weights = NULL;
        biases = NULL;
        in_features = in;
        out_features = out;
    }
    void init_func(FILE* weight_file){
        float* w = (float*)malloc(in_features * out_features * sizeof(float));
        float* b = (float*)malloc(out_features * sizeof(float));
        fread(w, sizeof(float), in_features * out_features, weight_file);
        fread(b, sizeof(float), out_features, weight_file);
        cudaMalloc((void**)&weights, in_features * out_features * sizeof(float)); // выделяет память на устройстве CUDA для весов
        cudaMalloc((void**)&biases, out_features * sizeof(float)); // выделяет память на устройстве CUDA для смещений
        cudaMemcpy(weights, w, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice); // копирует значения весов с хоста на устройство
        cudaMemcpy(biases, b, out_features * sizeof(float), cudaMemcpyHostToDevice); // копирует значения смещений с хоста на устройство
        free(w);
        free(b);
    }
    float* operator() (float* x) // оператор ()
    {
        const float alpha = 1;

        //CUBLAS_OP_T - транспонирование матрицы весов
        cublasSgemv(handle, CUBLAS_OP_T, in_features, out_features, &alpha, weights, in_features, x, 1, &alpha, biases, 1); // умножение входного массива на веса слоя и добавление смещений
        cublasScopy(handle, out_features, biases, 1, x, 1); // копирует результат входного массива обратно в x
        return x;
    }
    ~Linear() {
        if (weights)
            cudaFree(weights);
        if (biases)
            cudaFree(biases);
    }
};

class Net {
    float* forward(float* x) {
        sigmoid<<<1, 256>>>(fc1(x)); // 1 блок, и внутри него будет 256 потоков, каждый из которых выполняет вычисления для одного элемента входного массива x
        sigmoid<<<1, 16>>>(fc2(x));
        sigmoid<<<1, 1>>>(fc3(x));
        return x;
    }
public:
    Linear fc1;
    Linear fc2;
    Linear fc3;
    Net() {
        fc1 = Linear(1024, 256);
        fc2 = Linear(256, 64);
        fc3 = Linear(64, 1);
    }
    float* operator() (float* r) {
        return forward(r);
    }
};

int main() {
    int size = 1024;
    float* input_layer = (float*)malloc(size * sizeof(float));
    FILE* input_file = fopen("input.npy", "rb");
    FILE* weight_file = fopen("weight.npy", "rb");
    if (input_layer) fread(input_layer, sizeof(float), size, input_file);

    float* layer_gp;
    cudaMalloc((void**)&layer_gp, size); // выделяем память для входного слоя на устройстве CUDA
    cudaMemcpy(layer_gp, input_layer, size, cudaMemcpyHostToDevice); // копируем значения из хоста в устройство

    cublasCreate(&handle); // для использования функциональности библиотеки CUBLAS

    Net net = Net();
    net.fc1.init_func(weight_file);
    net.fc2.init_func(weight_file);
    net.fc3.init_func(weight_file);


    float itog;
    float* itog_gp;

    cudaMalloc((void**)&itog_gp, sizeof(float));
    itog_gp = net(layer_gp);
    cudaMemcpy(&itog, itog_gp, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%lf\n\n", itog);

    cublasDestroy(handle);
    free(input_layer);

    return 0;
}
