#include <iostream>
#include <cufft.h>
#include <malloc.h>
#include <fstream>

#define NX 365
#define BATCH 1

int main()
{
    cufftHandle plan;
    cufftComplex *data;
    auto *data_h = (cufftComplex *)calloc(NX * BATCH, sizeof(cufftComplex));

    std::ifstream fin("../data.txt");
    if(fin.fail())
    {
        std::cerr << "Failed to open file." << std::endl;
        return -1;
    }
    std::ofstream fout("../results.txt");
    float month, day, wolf, temp;
    for(int i = 0; i < NX; i++)
    {
        fin >> month >> day >> wolf >> temp;
        if(wolf != 999)
            data_h[i].x = wolf;
        else
            data_h[i].x = 0.0f;
        data_h[i].y = 0.0f;
    }

    cudaMalloc((void **)&data, sizeof(cufftComplex) * NX * BATCH);
    if(cudaGetLastError() != cudaSuccess) {
        std::cerr << "Cuda error: Failed to allocate." << std::endl;
        return -1;
    }
    cudaMemcpy(data, data_h,  sizeof(cufftComplex) * NX * BATCH, cudaMemcpyHostToDevice);
    if(cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: Plan creation failed." << std::endl;
        return -1;
    }
    if(cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error: ExecC2C Forward failed." << std::endl;
        return -1;
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "Cuda error: Failed to synchronize." << std::endl;
        return -1;
    }
    cudaMemcpy(data_h, data, NX * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for(int i = 0; i < NX; i++)
    {
        std::cout << data_h[i].x << "\t" << data_h[i].y << std::endl;
        fout << data_h[i].x << "\t" << data_h[i].y << std::endl;
    }

    cufftDestroy(plan);
    cudaFree(data);
    free(data_h);
    fin.close();
    fout.close();
    return 0;
}