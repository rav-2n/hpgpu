#include <cuda.h> 

class DataGeneratorWithCuda
{
public:
    __device__ void dataGen(uint64_t const limit, std::vector<uint64_t>* );
};