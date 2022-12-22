#include <random>
#include <iostream>
#include <array>
#include <vector>

#include "DataGenerator.h"


int main() {
    DataGenerator dataGenerator;
    uint64_t size = 1e4;
    std::vector<uint64_t> arr = dataGenerator.generatePseudoRandomNumbers<uint64_t>(size);
    assert(arr.size() == size);
}