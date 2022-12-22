#include <iostream>
#include <limits>

#include "../src/Aggregator/Aggregator.cc"
#include "../src/Validator/Validator.cc"


int main()
{
    Aggregator aggregator;

    auto const size = 1e4;
    // create a vector and fill in with value 1
    std::vector<uint64_t> data(size, 1);

    auto total = aggregator.multithreadAtomicAddition(data);

    auto const expectedTotal = 1e4;
    Validator validator;
    assert(validator.isEqualWithinTolerance(total, expectedTotal));
}