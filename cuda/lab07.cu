#include <cstdio>
#include <cmath>

#include <thrust/device_vector.h>

struct thrust_math {
    float u, t, h;

    thrust_math(float _u, float _t, float _h) : u(_u), t(_t), h(_h) {};

    __host__ __device__
    float operator()(const float &x, const float &y) const {
        return x + (y - x) * u * t / h;
    }
};

__global__
void kernel(float *x, float *y, float u, float t, float h) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    y[idx + 1] = x[idx + 1] + (x[idx] - x[idx + 1]) * u * t / h;
}

int main() {
    float finish;
    int n = 1024;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    thrust::host_vector<float> hostVector(n);
    thrust::device_vector<float> deviceVector(n);

    for (int i = 0; i < n; ++i)
        hostVector[i] = exp(-powf((i / 100.0 - 4.5), 2)) * 100 / (2 * sqrtf(2 * M_PI));
    thrust::copy(hostVector.begin(), hostVector.end(), deviceVector.begin());

    thrust::transform(thrust::device, deviceVector.begin() + 1, deviceVector.end(), deviceVector.begin(),
                      deviceVector.begin(), thrust_math(1.1, 0.9, 1.4));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&finish, start, stop);
    for (float i : hostVector) {
        printf("%.8f ", i);
    }
    printf("\n");
    for (float i : deviceVector) {
        printf("%.8f ", i);
    }
    printf("\n");

    printf("Thrust time: %f ms.\n", finish);

    cudaEventRecord(start, 0);
    float *hostArr, *hostArrRes, *cudaArr, *cudaArrRes;

    hostArr = (float *) malloc(n * sizeof(float));
    hostArrRes = (float *) malloc(n * sizeof(float));
    cudaMalloc((void **) &cudaArr, (n + 1) * sizeof(float));
    cudaMalloc((void **) &cudaArrRes, (n + 1) * sizeof(float));

    for (int i = 0; i < n; i++)
        hostArr[i] = exp(-powf((i / 100.0 - 4.5), 2)) * 100 / (2 * sqrtf(2 * M_PI));

    cudaMemcpy(cudaArr, hostArr, n * sizeof(float), cudaMemcpyHostToDevice);
    kernel <<<4, 256 >>>(cudaArr, cudaArrRes, 1.1, 0.9, 1.4);
    cudaMemcpy(hostArrRes, cudaArrRes + 1, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&finish, start, stop);
    printf("Native CUDA time: %f ms.", finish);
}
