#include <iostream>

#include "Validator.h"

int main() {
    long double const x = 1e10, y = 1e10 * 1.000001;

    Validator validator;
    assert(validator.isEqualWithinTolerance(x,y, 1e-5));
}