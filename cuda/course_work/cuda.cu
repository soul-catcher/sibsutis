#include <iostream>
#include <chrono>

const size_t size = 1 << 20;

__global__ void transpose(float_t *matrixOrigin, float_t *matrixRes) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t width = gridDim.x * blockDim.x;
    matrixRes[x + y * width] = matrixOrigin[y + x * width];
}

__global__ void saxpy(float_t *vectorA, float_t *vectorB, float_t alpha) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    vectorA[index] = vectorA[index] * alpha + vectorB[index];
}

int32_t main() {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    cudaStream_t stream0;
    const size_t num = 32;
    const size_t Nx = 1 << 10;
    const size_t Ny = 1 << 10;
    cudaStreamCreate(&stream0);
    float_t *matrix, *matrix_dev_origin, *matrix_dev_res;
    cudaHostAlloc((void **) &matrix, size * sizeof(float_t), cudaHostAllocDefault);
    for (int64_t i = 0; i < size; ++i)
        matrix[i] = i;
    cudaMalloc((void **) &matrix_dev_origin, sizeof(float_t) * size);
    cudaMalloc((void **) &matrix_dev_res, sizeof(float_t) * size);

    cudaMemcpyAsync(matrix_dev_origin, matrix, sizeof(float_t) * size, cudaMemcpyHostToDevice, stream0);
    transpose <<< dim3(Nx / num, Ny / num), dim3(num, num) >>>(matrix_dev_origin, matrix_dev_res);
    cudaMemcpyAsync(matrix, matrix_dev_res, sizeof(float_t) * size, cudaMemcpyDeviceToHost, stream0);
    cudaStreamSynchronize(stream0);
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double_t> time_span = std::chrono::duration_cast<std::chrono::duration<double_t>>(
            stop - start);
    std::cout << "Transpose time (s) - " << time_span.count() << std::endl;
//        for(int64_t i = 0; i < Ny; ++i)
//        {
//            for(int64_t j = 0; j < Nx; ++j)
//                std:: cout << matrix[i * Nx + j] << " ";
//            std::cout << std::endl;
//        }
    cudaFree(matrix_dev_origin);
    cudaFree(matrix_dev_res);
    cudaFreeHost(matrix);
    start = std::chrono::high_resolution_clock::now();
    float_t *vecA, *vecB, *vecA_device, *vecB_device;
    cudaStream_t stream_m0;
    cudaStreamCreate(&stream_m0);
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaHostAlloc((void **) &vecA, size * sizeof(float_t), cudaHostAllocDefault);
    cudaHostAlloc((void **) &vecB, size * sizeof(float_t), cudaHostAllocDefault);
    for (int64_t i = 0; i < size; ++i) {
        vecA[i] = i;
        vecB[i] = i * 2 - 1;
    }
    cudaMalloc((void **) &vecA_device, sizeof(float_t) * size);
    cudaMalloc((void **) &vecB_device, sizeof(float_t) * size);
    cudaMemcpyAsync(vecA_device, vecA, sizeof(int) * size, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(vecB_device, vecB, sizeof(int) * size, cudaMemcpyHostToDevice, stream1);
    saxpy <<< size / 2 / 1024, 1024, 0, stream0 >>>(vecA_device, vecB_device, 2.25);
    saxpy <<< size / 2 / 1024, 1024, 0, stream1 >>>(vecA_device + size / 2, vecB_device + size / 2, 2.25);
    cudaMemcpyAsync(vecA, vecA_device, sizeof(float_t) * size / 2, cudaMemcpyDeviceToDevice, stream0);
    cudaMemcpyAsync(vecA + size / 2, vecA_device + size / 2, sizeof(float_t) * size / 2, cudaMemcpyDeviceToDevice,
                    stream1);
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    stop = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double_t>>(stop - start);
    std::cout << "SAXPY time (s) - " << time_span.count() << std::endl;
    //for (int64_t i = 0; i < size; ++i)
    //	std::cout << vecA[i] << " ";
    cudaFree(vecA_device);
    cudaFree(vecB_device);
    cudaFreeHost(vecA);
    cudaFreeHost(vecB);
}