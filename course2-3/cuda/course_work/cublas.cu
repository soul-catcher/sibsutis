#include <cublas_v2.h>
#include <iostream>
#include <cstddef>
#include <iomanip>
#include <chrono>


int main()
{
    size_t cols = 1 << 10;
    size_t rows = 1 << 10;
    size_t N = cols * rows;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    cublasHandle_t handle;
    cublasCreate(&handle);
    float_t* matrix;
    cudaMallocHost((void**)&matrix, N * sizeof(float_t));
    for (int i = 0; i < N; ++i)
        matrix[i] = static_cast<float_t>(i);
    float_t* matrix_in_dev;
    cudaMalloc((void**)&matrix_in_dev, N * sizeof(float_t));
    float_t* matrix_out_dev;
    cudaMalloc((void**)&matrix_out_dev, N * sizeof(float_t));
    cublasSetMatrix(rows, cols, sizeof(float_t), matrix, rows, matrix_in_dev, rows);
    float_t alpha = 1.; // change sample
    float_t beta = 0.;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, cols, rows, &alpha, matrix_in_dev, rows, &beta, matrix_in_dev, rows, matrix_out_dev, cols);
    cublasGetMatrix(rows, cols, sizeof(float_t), matrix_out_dev, rows, matrix, rows);
    cudaStreamSynchronize(nullptr);
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double_t> time_span = std::chrono::duration_cast<std::chrono::duration<double_t>>(stop - start);
//       for (int i = 0; i < rows; ++i)
//        {
//            for (int j = 0; j < cols; ++j)
//                std::cout << std::setw(10) << static_cast<float>(matrix[j + i * rows]) << "\t";
//            std::cout << "\n";
//        }
    cudaFreeHost(matrix);
    cudaFree(matrix_in_dev);
    cudaFree(matrix_out_dev);
    cublasDestroy(handle);
    std::cout << "Matrix transpose (s) " << "  -  " << time_span.count() << "\n";
    cublasHandle_t handle1;
    cublasCreate(&handle1);

    start = std::chrono::high_resolution_clock::now();
    float_t* vecA;
    cudaMallocHost((void**)&vecA, N * sizeof(float_t));
    float_t* vecB;
    cudaMallocHost((void**)&vecB, N * sizeof(float_t));

    for (int i = 0; i < N; ++i)
    {
        vecA[i] = (float_t)i;
        vecB[i] = (float_t)(i * 2 - 1);
    }
    float_t* aDev;
    cudaMalloc((void**)&aDev, N * sizeof(float_t));
    float_t* bDev;
    cudaMalloc((void**)&bDev, N * sizeof(float_t));

    cublasSetMatrix(N, 1, sizeof(float_t), vecA, N, aDev, N);
    cublasSetMatrix(N, 1, sizeof(float_t), vecB, N, bDev, N);
    alpha = 2.25;
    cublasSaxpy(handle1, N, &alpha, aDev, 1, bDev, 1);
    cublasGetMatrix(N, 1, sizeof(float_t), bDev, N, vecB, N);
    cudaStreamSynchronize(nullptr);
    //for (int i = 0; i < N; ++i)
    //	printf("%f\n", vecB[i]);
    cublasDestroy(handle1);
    cudaFreeHost(vecA);
    cudaFreeHost(vecB);
    cudaFree(aDev);
    cudaFree(bDev);
    stop = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double_t>>(stop - start);
    std::cout << "Saxpy time (s) " << "  -  " << time_span.count() << "\n";
}