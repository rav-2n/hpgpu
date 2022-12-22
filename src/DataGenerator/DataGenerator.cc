#include <random>
#include <iostream>
#include <array>
#include <vector>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>

class DataGenerator
{
public:
    template <class DataType>
    std::vector<DataType> generatePseudoRandomNumbers(DataType const limit)
    {
        int const seed = 100;
        std::default_random_engine randomEngine(seed);

        std::uniform_int_distribution<DataType> dist(
            std::numeric_limits<DataType>::min(),
            std::numeric_limits<DataType>::max());

        using namespace indicators;
        show_console_cursor(false);

        ProgressBar bar{
            option::BarWidth{50},
            option::Start{"["},
            option::Fill{"-"},
            option::Lead{">"},
            option::Remainder{" "},
            option::End{"]"},
            option::PostfixText{"Generating Data"},
            option::ForegroundColor{Color::green},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

        std::vector<DataType> array;
        array.reserve(limit);
        for (int i = 0; i < limit; i++)
        {
            bar.set_progress(((double)i / (double)limit) * 100);
            bar.tick();

            auto elem = dist(randomEngine);
            array.push_back(elem);
            if (bar.is_completed())
                break;
        }

        std::cout << "Generated a vector with " << limit << " elements" << std::endl;
        std::cout << "Size: " << limit * sizeof(uint64_t) / 1e6 << "mb" << std::endl;
        show_console_cursor(true);

        return array;
    }
};
