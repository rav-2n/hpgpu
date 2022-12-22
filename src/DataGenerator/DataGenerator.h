#include <iostream>

class DataGenerator
{
public:
    template <class DataType>
    std::vector<DataType> generatePseudoRandomNumbers(uint64_t const);
};
