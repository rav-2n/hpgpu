#include <random>
#include <iostream>
#include <array>
#include <vector>

class DataGenerator
{
public:
    template <class DataType>
    std::vector<DataType> generatePseudoRandomNumbers(uint64_t const);
};
