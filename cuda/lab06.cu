#include <cstdio>


__global__ void sum(const int *a, const int *b, int *c)
{
    int elemInd = threadIdx.x + blockDim.x * blockIdx.x;
    c[elemInd] = a[elemInd] + b[elemInd];
}

__global__ void multiply(int *a, int *b, int *c)
{
    int elemInd = threadIdx.x + blockDim.x * blockIdx.x;
    a[elemInd] *= b[elemInd];
    int i = gridDim.x * blockDim.x / 2;
    __syncthreads();
    while (i)
    {
        if (elemInd < i)
            a[elemInd] += a[elemInd + i];
        __syncthreads();
        i /= 2;
    }
    if (elemInd == 0)
        *c = a[0];
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int size = 1 << 20;
    float elapsedTime;
    int *hostArr, *stream0Arr, *stream1Arr, *hostResult, *stream0Result, *stream1Result;
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaHostAlloc((void**)&hostArr, size * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostResult, size * sizeof(int), cudaHostAllocDefault);
    for (int i = 0; i < size; ++i)
        hostArr[i] = rand() % 100;

    printf("sumTime1\tsumTime2\n");
    for (int i = 256; i < (size >> 1); i <<= 1)
    {
        printf ("i = %d\n", i);
        cudaMalloc((void **) &stream0Arr, i * sizeof (int));
        cudaMalloc((void **) &stream0Result, i * sizeof (int));
        cudaMalloc((void **) &stream1Arr, i * sizeof (int));
        cudaMalloc((void **) &stream1Result, i * sizeof (int));
        //1
        cudaEventRecord (start, nullptr);
        for (int j = 0; j < size; j += 2 * i)
        {
            cudaMemcpyAsync (stream0Arr, hostArr + j, sizeof (int) * i, cudaMemcpyHostToDevice, stream0);
            sum <<<i / 256, 256, 0, stream0 >>> (stream0Arr, stream0Arr, stream0Result);
            cudaMemcpyAsync (hostResult + j, stream0Result, sizeof (int) * i, cudaMemcpyDeviceToHost, stream0);

            cudaMemcpyAsync (stream1Arr, hostArr + j + i, sizeof (int) * i, cudaMemcpyHostToDevice, stream1);
            sum <<<i / 256, 256, 0, stream1>>> (stream1Arr, stream1Arr, stream1Result);
            cudaMemcpyAsync (hostResult + j + i, stream1Result, sizeof (int) * i, cudaMemcpyDeviceToHost, stream1);
        }
        cudaStreamSynchronize (stream0);
        cudaStreamSynchronize (stream1);
        cudaEventRecord (stop, nullptr);
        cudaEventSynchronize (stop);
        cudaEventElapsedTime (&elapsedTime, start, stop);
        printf ("%f\t", elapsedTime);

        memset (hostResult, 0, size * sizeof (int));
        //2
        cudaEventRecord (start, nullptr);
        for (int j = 0; j < size; j += 2 * i)
        {
            cudaMemcpyAsync (stream0Arr, hostArr + j, sizeof (int) * i, cudaMemcpyHostToDevice, stream0);
            cudaMemcpyAsync (stream0Arr, hostArr + j + i, sizeof (int) * i, cudaMemcpyHostToDevice, stream1);
            sum <<<i / 256, 256, 0, stream0 >>> (stream0Arr, stream0Arr, stream0Result);
            sum <<<i / 256, 256, 0, stream1 >>> (stream1Arr, stream1Arr, stream1Result);
            cudaMemcpyAsync (hostResult + j, stream0Result, sizeof (int) * i, cudaMemcpyDeviceToHost, stream0);
            cudaMemcpyAsync (hostResult + j + i, stream1Result, sizeof (int) * i, cudaMemcpyDeviceToHost, stream1);
        }
        cudaStreamSynchronize (stream0);
        cudaStreamSynchronize (stream1);
        cudaEventRecord (stop, nullptr);
        cudaEventSynchronize (stop);
        cudaEventElapsedTime (&elapsedTime, start, stop);
        printf ("%f\n", elapsedTime);

        memset(hostResult, 0, size * sizeof(int));

        cudaFree(stream0Arr);
        cudaFree(stream0Result);
        cudaFree(stream1Arr);
        cudaFree(stream1Result);
    }

    printf("multiplyTime1\tmultiplyTime2\n");
    for (int i = 256; i < (size >> 1); i <<= 1)
    {
        printf ("i = %d\n", i);
        cudaMalloc ((void **) &stream0Arr, i * sizeof (int));
        cudaMalloc ((void **) &stream0Result, i * sizeof (int));
        cudaMalloc ((void **) &stream1Arr, i * sizeof (int));
        cudaMalloc ((void **) &stream1Result, i * sizeof (int));
        //1
        cudaEventRecord(start, 0);
        for (int j = 0; j < size; j += 2 * i)
        {
            cudaMemcpyAsync(stream0Arr, hostArr + j, sizeof(int) * i, cudaMemcpyHostToDevice, stream0);
            multiply <<<i / 256, 256, 0, stream0 >>> (stream0Arr, stream0Arr, stream0Result);
            cudaMemcpyAsync(hostResult + j, stream0Result, sizeof(int) * i, cudaMemcpyDeviceToDevice, stream0);

            cudaMemcpyAsync(stream1Arr, hostArr + j + i, sizeof(int) * i, cudaMemcpyHostToDevice, stream1);
            multiply <<<i / 256, 256, 0, stream1 >>> (stream1Arr, stream1Arr, stream1Result);
            cudaMemcpyAsync(hostResult + j + i, stream1Result, sizeof(int) * i, cudaMemcpyDeviceToDevice, stream1);
        }
        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("%f\t", elapsedTime);

        memset(hostResult, 0, size * sizeof(int));
        //2
        cudaEventRecord(start, 0);
        for (int j = 0; j < size; j += 2 * i)
        {
            cudaMemcpyAsync(stream0Arr, hostArr + j, sizeof(int) * i, cudaMemcpyHostToDevice, stream0);
            cudaMemcpyAsync(stream1Arr, hostArr + j + i, sizeof(int) * i, cudaMemcpyHostToDevice, stream1);
            multiply <<<i / 256, 256, 0, stream0 >>> (stream0Arr, stream0Arr, stream0Result);
            multiply <<<i / 256, 256, 0, stream1 >>> (stream1Arr, stream1Arr, stream1Result);
            cudaMemcpyAsync(hostResult + j, stream0Result, sizeof(int) * i, cudaMemcpyDeviceToDevice, stream0);
            cudaMemcpyAsync(hostResult + j + i, stream1Result, sizeof(int) * i, cudaMemcpyDeviceToDevice, stream1);
        }
        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("%f\n", elapsedTime);

        memset(hostResult, 0, size * sizeof(int));

        cudaFree(stream0Arr);
        cudaFree(stream0Result);
        cudaFree(stream1Arr);
        cudaFree(stream1Result);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(hostArr);
    cudaFreeHost(hostResult);
    return 0;
}