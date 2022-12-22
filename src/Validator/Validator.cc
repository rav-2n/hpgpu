#include <iostream>
#include <limits>

class Validator
{
public:
    bool isEqualWithinTolerance(long double const x, long double const y,
                              long double const tolerance = 1e-6, long double zeroTolerance = 1e-30)
    {
        auto threshold = std::numeric_limits<long double>::min();
        auto min = std::min(std::abs(x), std::abs(y));
        if (std::abs(min) == 0.0)
        {
            return std::abs(x - y) < zeroTolerance;
        }
        return (std::abs(x - y) / std::max(threshold, min)) < tolerance;
    }
};

int main() {
    long double const x = 1e10, y = 1e10 * 1.000001;

    Validator validator;
    assert(validator.isEqualWithinTolerance(x,y, 1e-5));
}