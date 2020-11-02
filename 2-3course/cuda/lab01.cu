#include <iostream>
#include <chrono>
#include <cmath>

__global__ void vectors_add(float arr1[], float arr2[]) {
    arr1[threadIdx.x + blockDim.x * blockIdx.x] += arr2[threadIdx.x + blockDim.x * blockIdx.x];
}

int main(int argc, char *argv[]) {
    float *arr1, *arr2, *res, *devarr1, *devarr2;
    for (int N = pow(2, 28), i = 0; i < 19; N /= 2, i++) {
        int threads_per_block = std::stoi(argv[1]);
        int num_of_blocks = N / threads_per_block;

        arr1 = new float[N];
        arr2 = new float[N];
        res = new float[N];
        for (int i = 0; i < N; i++) {
            arr1[i] = (float) rand() / RAND_MAX;
            arr2[i] = (float) rand() / RAND_MAX;
        }
        cudaError_t errcode = cudaMalloc((void **) &devarr1, N * sizeof(float));
        if (errcode != cudaSuccess) {
            std::cout << "Error" << std::endl;
            return 0;
        }
        errcode = cudaMalloc((void **) &devarr2, N * sizeof(float));
        if (errcode != cudaSuccess) {
            std::cout << "Error" << std::endl;
            return 0;
        }
        errcode = cudaMemcpy(devarr1, arr1, N * sizeof(float), cudaMemcpyHostToDevice);
        if (errcode != cudaSuccess) {
            std::cout << "Error" << std::endl;
            return 0;
        }
        errcode = cudaMemcpy(devarr2, arr2, N * sizeof(float), cudaMemcpyHostToDevice);
        if (errcode != cudaSuccess) {
            std::cout << "Error" << std::endl;
            return 0;
        }
        auto start = std::chrono::high_resolution_clock::now();
        vectors_add<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(devarr1, devarr2);
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << duration.count() << std::endl;
        cudaMemcpy(res, devarr1, N * sizeof(float), cudaMemcpyDeviceToHost);
        //for(int i = 0; i < 100; i++)
        //	std::cout << res[i] << '\n';
        for (int i = 0; i < N; i++) {
            if (res[i] != arr1[i] + arr2[i]) {
                std::cout << "Error" << std::endl;
                return 0;
            }
        }
        cudaFree(devarr1);
        cudaFree(devarr2);
        delete[] arr1;
        delete[] arr2;
        delete[] res;
    }
}
