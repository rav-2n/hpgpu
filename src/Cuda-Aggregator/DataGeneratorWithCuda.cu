#include <iostream>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>
#include <limits>

#include "cuda_helper.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

using std::cout;
using std::endl;

__global__ void callKernel(int, uint64_t *);

class DataGeneratorWithCuda
{
public:
    __device__ void dataGen(uint64_t const limit, uint64_t *vec)
    {
        winsize w;
        // ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

        auto const globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint64_t i = globalThreadId; i < limit; i += gridDim.x * blockDim.x)
        {
            vec[i] = (uint64_t)i;

            auto const percent = double(i) / double(limit) * 100;
            auto const percentFloat = double(i) / double(limit);
            // cout << '\r' << std::left << "[" << std::flush;
            // cout << '\r' << std::setw(w.ws_col * percentFloat * 0.8) << std::setfill('-') << std::flush;
            // cout << '\r' << std::right << "] " << percent << "%" << std::flush;
        }
        // cout << endl;
    }

    template <int NBlock, int NThread>
    void gen(uint64_t const limit)
    {
        // int NBlock = 1;
        // int NThread = 128;
        uint64_t vec_h[limit];
        uint64_t *vec_d;
        // time measurement
        cudaEvent_t cstart, cend;
        CHECK_CUDA(cudaEventCreate(&cstart));
        CHECK_CUDA(cudaEventCreate(&cend));
        float milliseconds = 0;
        float min_ms = std::numeric_limits<float>::max();

        CHECK_CUDA(cudaMalloc((void **)&vec_d, limit * sizeof(uint64_t)));

        CHECK_CUDA(cudaEventRecord(cstart));

        // callKernel<<<NBlock, NThread>>>(limit, vec_d);
        CUDA_CHECK_KERNEL;

        // CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(cend));
        CHECK_CUDA(cudaEventSynchronize(cend));

        CHECK_CUDA(cudaMemcpy(vec_h, vec_d, sizeof(uint64_t) * limit, cudaMemcpyDeviceToHost));

        // for (int i = 0; i < limit; i++)
        // {
        //     cout << vec_h[i] << endl;
        // }

        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, cstart, cend));
        if (milliseconds < min_ms)
            min_ms = milliseconds;

        // output
        cout
            << "GPU: (min kernels time = " << min_ms << " ms)\n"
            << "max bandwidth: " << limit * sizeof(uint64_t) / min_ms * 1e-6 << " GB/s\n"
            << "Worked with array of size: " << sizeof(vec_h) / (double)1e6 << " Mb"
            << endl;

        // delete[] vec_h;
        CHECK_CUDA(cudaFree(vec_d));
        CHECK_CUDA(cudaEventDestroy(cstart));
        CHECK_CUDA(cudaEventDestroy(cend));
    }
};

__global__ void callKernel(int limit, uint64_t *vec)
{
    DataGeneratorWithCuda dataGeneratorWithCuda;
    dataGeneratorWithCuda.dataGen(limit, vec);
}

int main()
{
    DataGeneratorWithCuda dataGeneratorWithCuda;
    int const NBlock = 26;
    int const NThread = 1024;

    uint64_t const limit = 1e8;

    dataGeneratorWithCuda.gen<NBlock, NThread>(limit);
    return 0;
}