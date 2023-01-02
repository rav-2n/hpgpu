#include "./cuda_helper.cuh"
#include "./DataGeneratorWithCuda.cu"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <limits>

template <typename T, int TBlocksize>
__global__ void aggregate_addition_with_shared_memory(T *, T *, int);

class CudaAggregatorSimple
{

public:
    int addition_with_shared_memory(uint64_t size, uint64_t init)
    {
        int device = 0;

        try
        {
            aggregate_addition<uint64_t, 128>(size, device, init);
        }
        catch (std::runtime_error &error)
        {
            std::cerr << error.what() << "\n";
            CHECK_CUDA(cudaDeviceReset());
            return 1;
        }
        CHECK_CUDA(cudaDeviceReset());
        return 0;
    }

private:
    template <typename T, int TBlocksize>
    void aggregate_addition(uint64_t n, int dev, T init = 0)
    {

        CHECK_CUDA(cudaSetDevice(dev));
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
        cudaEvent_t cstart, cend;
        CHECK_CUDA(cudaEventCreate(&cstart));
        CHECK_CUDA(cudaEventCreate(&cend));

        std::cout << getCUDADeviceInformations(dev).str()
                  << "\n\n";

        int numSMs;
        CHECK_CUDA(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
        dim3 blocks(16 * numSMs); // 16*128 = 2048 (=max resident threads on SM), rule of thumb
        if (blocks.x > ((n - 1) / TBlocksize + 1))
            blocks.x = (n - 1) / TBlocksize + 1;

        // T *h_x = new T[n];
        T h_x[n];
        T *x = nullptr;
        T *y = nullptr;
        T result_gpu = 0;

        CHECK_CUDA(cudaMalloc(&x, n * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&y, blocks.x * sizeof(T)));

        // init host memory - TODO: USE DATA GENERATOR AFTER INTRODUCING CMAKE
        if (init != 0)
        {
            for (int i = 0; i < n; i++)
                h_x[i] = init;
        }

        else
        {
            int const NBlock = 26;
            int const NThread = 1024;
            DataGeneratorWithCuda::gen<NBlock, NThread>(n, h_x);
        }

        CHECK_CUDA(cudaMemcpy(x, h_x, n * sizeof(T), cudaMemcpyHostToDevice));

        float milliseconds = 0;
        float min_ms = std::numeric_limits<float>::max();

        CHECK_CUDA(cudaMemset(y, 0, sizeof(T)));
        CHECK_CUDA(cudaEventRecord(cstart));

        aggregate_addition_with_shared_memory<uint64_t, TBlocksize><<<blocks, TBlocksize>>>(x, y, n);
        aggregate_addition_with_shared_memory<uint64_t, TBlocksize><<<1, TBlocksize>>>(y, y, blocks.x);

        CHECK_CUDA(cudaEventRecord(cend));
        CHECK_CUDA(cudaEventSynchronize(cend));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, cstart, cend));
        if (milliseconds < min_ms)
            min_ms = milliseconds;

        CHECK_CUDA(cudaMemcpy(&result_gpu, y, sizeof(T), cudaMemcpyDeviceToHost));

        std::cout << "Result (n = " << n << "):\n"
                  << "GPU: " << result_gpu << " (min kernels time = " << min_ms << " ms)\n";

        if (init != 0)
        {
            cout << "expected: " << init * n << "\n"
                 << (init * n != result_gpu ? "MISMATCH!!" : "Success") << "\n";
        }

        cout << "max bandwidth: " << n * sizeof(T) / min_ms * 1e-6 << " GB/s"
        << std::endl;

        // delete[] h_x;
        CHECK_CUDA(cudaFree(x));
        CHECK_CUDA(cudaFree(y));
        CHECK_CUDA(cudaEventDestroy(cstart));
        CHECK_CUDA(cudaEventDestroy(cend));
    }
};

template <typename T, int TBlocksize>
__global__ void aggregate_addition_with_shared_memory(T *x, T *y, int n)
{
    __shared__ T sdata[TBlocksize];

    int tid = threadIdx.x;
    int i = blockIdx.x * TBlocksize + threadIdx.x;

    // safeguard
    if (i > n)
        return;

    // store thread local sum in register, initialize with "current value"
    T tsum = x[i];

    // offset for each thread
    int gridsize = gridDim.x * TBlocksize;

    i += gridsize;

    // grid reduce
    while (i < n)
    {
        tsum += x[i];
        i += gridsize;
    }

    sdata[tid] = tsum;

    __syncthreads();

#pragma unroll
    for (unsigned int bs = TBlocksize,
                      bsup = (TBlocksize + 1) / 2; // ceil(TBlocksize/2.0)
         bs > 1;
         bs = bs / 2,
                      bsup = (bs + 1) / 2) // ceil(bs/2.0)
    {
        bool cond = threadIdx.x < bsup                                     // only first half of block is working
                    && (threadIdx.x + bsup) < TBlocksize                   // index for second half must be in bounds
                    && (blockIdx.x * TBlocksize + threadIdx.x + bsup) < n; // if elem in second half has been initialized before
        if (cond)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + bsup];
        }
        __syncthreads();
    }

    __syncthreads();

    if (tid == 0)
        y[blockIdx.x] = sdata[0];
}

int main()
{
    CudaAggregatorSimple cudaAggregatorSimple;
    cudaAggregatorSimple.addition_with_shared_memory(1e8, 1);
    return 0;
}