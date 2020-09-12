#include <iostream>

#include <cublas_v2.h>
#include <thrust/device_vector.h>

struct saxpy_functor
{
    const float a_;
    saxpy_functor(float a) : a_(a) {}
    __host__ __device__
    float operator() (const float &x, const float &y) const {
        return a_ * x + y;
    }
};

int main ()
{
    cudaEvent_t start, finish;
    cudaEventCreate(&start);
    cudaEventCreate(&finish);
    float elapsedTime;
    const int N = 1 << 10;
    const float XVAL = rand() % 1000000;
    const float YVAL = rand() % 1000000;
    const float AVAL = rand() % 1000000;

    float *host_x, *host_y;

    cublasHandle_t handle;
    cublasCreate(&handle);

    host_x = new float[N];
    host_y = new float[N];
    for (int i = 0; i < N; i++) {
        host_x[i] = XVAL;
        host_y[i] = YVAL;
    }

    float *dev_x, *dev_y;
    cudaMalloc((void **) &dev_x, N * sizeof(float));
    cudaMalloc((void **) &dev_y, N * sizeof(float));
    cublasSetVector(N, sizeof(host_x[0]), host_x, 1, dev_x, 1);
    cublasSetVector(N, sizeof(host_y[0]), host_x, 1, dev_y, 1);
    cudaDeviceSynchronize();

    cudaEventRecord(start, nullptr);
    cublasSaxpy(handle, N, &AVAL, dev_x, 1, dev_y, 1);
    cudaDeviceSynchronize();
    cudaEventRecord(finish, nullptr);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(&elapsedTime, start, finish);
    std::cout << "CUBLAS SAXPY: " << elapsedTime << "ms.\n";

    cudaFree(dev_x);
    cudaFree(dev_y);
    delete [] host_x;
    delete [] host_y;

    thrust::device_vector<float> X(N, XVAL);
    thrust::device_vector<float> Y(N, YVAL);

    cudaEventRecord(start, nullptr);
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(AVAL));
    cudaDeviceSynchronize();
    cudaEventRecord(finish, nullptr);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(&elapsedTime, start, finish);
    std::cout << "Thrust SAXPY: " << elapsedTime << "ms.\n";
}
