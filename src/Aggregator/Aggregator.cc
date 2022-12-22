#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <future>
#include <numeric>
#include <atomic>
#include <thread>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>

#include "../DataGenerator/DataGenerator.cc"
#include "../Logger/Logger.cc"

typedef std::chrono::high_resolution_clock Clock;

std::atomic<uint64_t> atomicAddition(0);

template <class DataType>
void add(std::vector<DataType> const &vec, uint64_t start, uint64_t stop)
{
    using namespace indicators;
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"-"},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Adding Elements"},
        option::ForegroundColor{Color::green},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true}};

    uint64_t progressBarStartIndex = 0;
    const uint64_t progressBarEndIndex = stop - start;
    for (uint64_t i = start; i < stop; i++)
    {
        progressBarStartIndex++;
        bar.set_progress(((double)progressBarStartIndex / (double)progressBarEndIndex) * 100);
        std::cout << "\x1B[2J\x1B[H";
        bar.tick();
        atomicAddition += vec[i];
    }
}

class Aggregator
{
public:
    template <class DataType>
    uint64_t multithreadAtomicAddition(std::vector<DataType> const &vec)
    {
        auto const t1 = Clock::now();



        std::thread thread1(add<DataType>, vec, 0, vec.size() / 4);
        std::thread thread2(add<DataType>, vec, vec.size() / 4, vec.size() / 2);
        std::thread thread3(add<DataType>, vec, vec.size() / 2, 3 * vec.size() / 4);
        std::thread thread4(add<DataType>, vec, 3 * vec.size() / 4, vec.size());

        thread1.join();
        thread2.join();
        thread3.join();
        thread4.join();
        auto total = atomicAddition.load();

        std::cout << "Sum of " << vec.size() << " elements: " << total << std::endl;

        Logger logger;
        logger.logDuration(t1, Clock::now());

        return total;
    }
};

// int main()
// {
//     auto size = 1e4;
//     DataGenerator dataGenerator;
//     std::vector<uint64_t> data = dataGenerator.generatePseudoRandomNumbers<uint64_t>(size);

//     Aggregator aggregator;
//     aggregator.multithreadAtomicAddition<uint64_t>(data);
// }