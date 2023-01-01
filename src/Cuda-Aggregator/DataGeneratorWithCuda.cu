#include <iostream>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

using std::cout;
using std::endl;

class DataGeneratorWithCuda 
{
public:
    __device__ void dataGen(uint64_t const limit, std::vector<uint64_t>* vec)
    {
        winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

        auto const globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (auto i = globalThreadId; i < limit; i += gridDim.x * blockDim.x)
        {
            vec[i] = globalThreadId;

            auto const percent = double(i) / double(limit) * 100;
            auto const percentFloat = double(i) / double(limit);
            cout << '\r' << std::left << "[" << std::flush;
            cout << '\r' << std::setw(w.ws_col * percentFloat * 0.8) << std::setfill('-') << std::flush;
            cout << '\r' << std::right << "] " << percent << "%" << std::flush;
        }
        cout << endl;
    }
}
