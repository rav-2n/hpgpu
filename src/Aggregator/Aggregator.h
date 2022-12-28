#include <iostream>
#include <array>
#include <vector>

class Aggragator
{
public:
    template <class DataType>
    uint64_t multithreadAtomicAddition(std::vector<DataType> const&);

    template <class DataType>
    uint64_t max(std::vector<DataType> const&);

    template <class DataType>
    uint64_t min(std::vector<DataType> const&);

    template <class DataType>
    uint64_t standardDeviation(std::vector<DataType> const&);

    template <class DataType>
    uint64_t variance(std::vector<DataType> const&);
};
