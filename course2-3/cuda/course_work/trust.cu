#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <iostream>

struct fun
{
    float alpha;
    explicit fun(float _alpha): alpha(_alpha) {}
    __host__ __device__
    float operator()(float x, float y) { return x * alpha + y; }
};

int main()
{
    size_t rows = 1 << 10;
    size_t cols = 1 << 10;
    size_t size_n = cols * rows;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<float_t> matrix_origin_dev(size_n);
    thrust::device_vector<float_t> matrix_res_dev(size_n);
    thrust::sequence(matrix_origin_dev.begin(), matrix_origin_dev.end());
    thrust::counting_iterator<size_t> indices(0);
    thrust::device_vector<float_t> temp(cols);
    for(size_t i = 0; i < rows; ++i)
    {
        thrust::sequence(temp.begin(), temp.end(), 0 + i, rows);
        thrust::copy(thrust::make_permutation_iterator(matrix_origin_dev.begin(), temp.begin()),
                     thrust::make_permutation_iterator(matrix_origin_dev.begin(), temp.end()), matrix_res_dev.begin() + i * cols);
    }
    thrust::host_vector<float_t> host_result = matrix_res_dev;
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double_t> time_span = std::chrono::duration_cast<std::chrono::duration<double_t>>(stop - start);
    std::cout << "Transpose time (s) - " << time_span.count() << std::endl;
//    for(int64_t i = 0; i < rows; ++i)
//    {
//        for(int64_t j = 0; j < cols; ++j)
//            std:: cout << host_result[i * rows + j] << " ";
//        std::cout << std::endl;
//    }
    start = std::chrono::high_resolution_clock::now();
    fun alpha(2.25);
    thrust::device_vector<float_t> vectorA(size_n);
    thrust::device_vector<float_t> vectorB(size_n);
    thrust::sequence(vectorA.begin(), vectorA.end());
    thrust::sequence(vectorB.begin(), vectorB.end(), -1, 2);
    thrust::transform(vectorA.begin(), vectorA.end(), vectorB.begin(), vectorA.begin(), alpha);
    thrust::host_vector<float_t> resVector = vectorA;
    stop = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double_t>>(stop - start);
    std::cout << "SAXPY time (s) - " << time_span.count() << std::endl;
//    for(auto &i : resVector)
//    {
//        std::cout << i << " ";
//    }
}
