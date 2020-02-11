#include <iostream>
#include <chrono>
#include <cmath>

#define CUDA_CHECK_RETURN(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);\
    }\
}
__global__ void vectors_add(float arr1[], float arr2[]){
        arr1[threadIdx.x+blockDim.x*blockIdx.x] += arr2[threadIdx.x+blockDim.x*blockIdx.x];
}

int main(int argc, char *argv[]){
    float *arr1, *arr2, *res, *devarr1, *devarr2;
    for (int threads_per_block = 1; threads_per_block <= 1024; threads_per_block *= 2) {
    std::cout << "threads per block:" << threads_per_block << std::endl;
        for (int N = pow(2, 28), i = 0; i < 19; N /= 2, i++) {
            int num_of_blocks = N / threads_per_block;         
            arr1 = new float[N];
            arr2 = new float[N];
            res = new float[N];
            for (int i = 0; i < N; i++) {
                arr1[i] = (float)rand() / RAND_MAX;
                arr2[i] = (float)rand() / RAND_MAX;
            }
            CUDA_CHECK_RETURN(cudaMalloc((void**)&devarr1, N * sizeof(float)));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&devarr2, N * sizeof(float)));
            CUDA_CHECK_RETURN(cudaMemcpy(devarr1, arr1, N*sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(devarr2, arr2, N*sizeof(float), cudaMemcpyHostToDevice));
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
            vectors_add<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(devarr1, devarr2);
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            CUDA_CHECK_RETURN(cudaGetLastError());
            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime,start,stop);
            std::cout << static_cast<int>(elapsedTime * 1000) << std::endl;
            CUDA_CHECK_RETURN(cudaMemcpy(res, devarr1, N*sizeof(float), cudaMemcpyDeviceToHost));
            //for(int i = 0; i < 100; i++)
            //  std::cout << res[i] << '\n';
            for (int i = 0; i < N; i++) {
                if (res[i] != arr1[i] + arr2[i]) {
                    std::cout << "Error" << std::endl;
                    return 0;
                }
            }
            CUDA_CHECK_RETURN(cudaFree(devarr1));
            CUDA_CHECK_RETURN(cudaFree(devarr2));
            delete[] arr1;
            delete[] arr2;
            delete[] res;
        }
    }
}
