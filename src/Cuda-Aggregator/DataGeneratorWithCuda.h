class DataGeneratorWithCuda
{
public:
    void dataGen(uint64_t const, uint64_t*);

    template <int NBlock, int NThread>
    static void gen(uint64_t, uint64_t*);
};