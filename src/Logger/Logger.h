#include <iostream>
#include <chrono>

typedef std::chrono::steady_clock::time_point time_point;

class Logger {
    public:
        void logDuration(time_point, time_point);
};