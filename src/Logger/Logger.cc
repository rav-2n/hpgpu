#include <iostream>
#include <chrono>

typedef std::chrono::steady_clock::time_point time_point;

class Logger {
    public:
        void logDuration(time_point t1, time_point t2) {
            auto const nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            auto const milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            auto const seconds = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

            std::cout << "Duration: " << nanoseconds << " nanaoseconds " << std::endl;
            if (milliseconds != 0) std::cout << " milliseconds " << std::endl;
            if (seconds != 0) std::cout << "Duration: " << seconds << " seconds " << std::endl;
        }
};