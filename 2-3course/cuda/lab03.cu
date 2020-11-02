#include <iostream>
#include <cmath>

#define CUDA_CHECK_RETURN(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);\
    }\
}

using namespace std;

__global__ void matrixInitByX(float *matrix) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    auto N = blockDim.x * gridDim.x;
    matrix[x + y * N] = (float) (x + y * N);
}

__global__ void matrixInitByY(float matrix[]) {
    auto y = threadIdx.x + blockIdx.x * blockDim.x;
    auto x = threadIdx.y + blockIdx.y * blockDim.y;
    auto N = blockDim.x * gridDim.x;
    matrix[x + y * N] = (float) (x + y * N);
}

__global__ void matrixTranspose(const float storage_d[], float storage_d_t[]) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    auto N = blockDim.x * gridDim.x;
    storage_d_t[j + i * N] = storage_d[i + j * N];
}


int main() {
    auto N = 1u << 8u;
    auto threads = 32;
    auto blocks = N / threads;
    float *matrix, *t_matrix;
    CUDA_CHECK_RETURN(cudaMallocManaged(&matrix, N * sizeof(float)))
    CUDA_CHECK_RETURN(cudaMallocManaged(&t_matrix, N * sizeof(float)))
    matrixInitByY<<<dim3(blocks, blocks), dim3(threads, threads)>>>(matrix);
    cudaDeviceSynchronize();
    CUDA_CHECK_RETURN(cudaGetLastError())

//    matrixTranspose<<<dim3(blocks, blocks), dim3(threads, threads)>>>(matrix, t_matrix);
//    cudaDeviceSynchronize();
//    CUDA_CHECK_RETURN(cudaGetLastError())
//    int side = static_cast<int>(sqrt(N));
//    for (int i = 0; i < side; i++) {
//        for (int j = 0; j < side; j++) {
//            cout << setw(3) << matrix[i * side + j] << ' ';
//        }
//        cout << '\n';
//    }
//    cout << '\n';
//    for (int i = 0; i < side; i++) {
//        for (int j = 0; j < side; j++) {
//            cout << setw(3) << t_matrix[i * side + j] << ' ';
//        }
//        cout << '\n';
//    }

}