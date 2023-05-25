#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>


cublasHandle_t handle;

__global__ void sigmoid(float* x) {
    int idx = threadIdx.x;
    x[idx] = exp(x[idx]) / (1 + exp(x[idx]));
}

class Linear {
    float* weight;
    float* bias;
    int in_features;
    int out_features;
public:
    Linear() {
        weight = NULL;
        bias = NULL;
        in_features = 0;
        out_features = 0;
    };
    Linear(int in, int out) {
        weight = NULL;
        bias = NULL;
        in_features = in;
        out_features = out;
    }
    void initializer(FILE* weights){
        float* w = (float*)malloc(in_features * out_features * sizeof(float));
        float* b = (float*)malloc(out_features * sizeof(float));
        fread(w, sizeof(float), in_features*out_features, weights);
        fread(b, sizeof(float), out_features, weights);
        cudaMalloc((void**)&weight, in_features * out_features * sizeof(float));
        cudaMalloc((void**)&bias, out_features * sizeof(float));
        cudaMemcpy(weight, w, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias, b, out_features * sizeof(float), cudaMemcpyHostToDevice);
        free(w);
        free(b);
    }
    float* operator() (float* x) {
        const float a = 1;
        cublasSgemv(handle, CUBLAS_OP_T, in_features, out_features, &a, weight, in_features, x, 1, &a, bias, 1);
        cublasScopy(handle, out_features, bias, 1, x, 1);
        return x;
    }
    ~Linear() {
        if (weight)
            cudaFree(weight);
        if (bias)
            cudaFree(bias);
    }
};

class Net {
    //прямое распространение информации
    float* forward(float* x) {
        sigmoid<<<1, 256>>>(fc1(x));
        sigmoid<<<1, 16>>>(fc2(x));
        sigmoid<<<1, 1>>>(fc3(x));
        return x;
    }
public:
    Linear fc1;
    Linear fc2;
    Linear fc3;
    Net() {
        fc1 = Linear((int)pow(32,2), (int)pow(16,2));
        fc2 = Linear((int)pow(16,2), (int)pow(4,2));
        fc3 = Linear((int)pow(4,2), 1);
    }
    float* operator() (float* r) {
        return forward(r);
    }
};

int main() {
    int size = 1024;
    float* input_layer = (float*)malloc(size * sizeof(float));
    FILE* input = fopen("input.npy", "rb");
    FILE* weight = fopen("weight.npy", "rb");
    if(input_layer) fread(input_layer, sizeof(float), size, input);

    float* d_layer;
    cudaMalloc((void**)&d_layer, size);
    cudaMemcpy(d_layer, input_layer, size, cudaMemcpyHostToDevice);

    cublasCreate(&handle);

    Net net = Net();
    net.fc1.initializer(weight);
    net.fc2.initializer(weight);
    net.fc3.initializer(weight);
    float result;
    float* d_res;
    cudaMalloc((void**)&d_res, sizeof(float));
    d_res = net(d_layer);
    cudaMemcpy(&result, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%lf\n\n", result);

    cublasDestroy(handle);

    free(input_layer);
    return 0;
}

