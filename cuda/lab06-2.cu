#include <cstdio>

void copying(float *h_a, float *h_b, float *d, unsigned int n, char *desc)
{
    printf("\n%s transfer\n", desc);
    unsigned int bytes = n * sizeof(float);
    float finishTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);
    cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&finishTime, start, stop);
    printf("Host to Device time: %f\n", finishTime);
    printf("Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / finishTime);

    cudaEventRecord(start, nullptr);
    cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&finishTime, start, stop);
    printf("Device to Host time: %f\n", finishTime);
    printf("Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / finishTime);

    for (int i = 0; i < n; i++)
    {
        if (h_a[i] != h_b[i])
        {
            printf("Smth failed :C\n");
            break;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{
    const int size = 4 * 1024 * 1024;
    const unsigned int bytes = size * sizeof(float);
    float *h_aPagable, *h_bPagable;
    float *h_aPinned, *h_bPinned;
    float *device;

    h_aPagable = (float *)malloc(bytes);
    h_bPagable = (float *)malloc(bytes);
    cudaMallocHost((void **)&h_aPinned, bytes);
    cudaMallocHost((void **)&h_bPinned, bytes);
    cudaMalloc((void **)&device, bytes);

    for (int i = 0; i < size; i++)
        h_bPagable[i] = i;
    memcpy(h_aPinned, h_aPagable, bytes);
    memset(h_bPagable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    copying(h_aPagable, h_bPagable, device, size, "Pageable");
    copying(h_aPinned, h_bPinned, device, size, "Pinned");

    cudaFree(device);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPagable);
    free(h_bPagable);

    return 0;
}