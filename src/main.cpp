#include "license_hdr.hpp"

#include "base_includes.hpp"
#include "generators.hpp"
#include "perf.hpp"
#include "module.hpp"
#include "simple.hpp"
#include "simple_4_arith.hpp"
#include "simple_cuda.hpp"

int main() {
    c_perf("Total Run time");
    srand(time(nullptr));

    constexpr auto wanted_highest_sym = 10000;
    size_t actual_highest_sym;
    std::vector<uint32_t> samples,samples2,samples3;
    std::vector<pair_t<uint32_t, uint32_t>> output,output2,output3;
    std::unique_ptr<c_data_generator_if<uint32_t>> gen (new c_data_generator_random_impl<uint32_t>());
    
    {
        c_perf("Generating Samples");
        gen->generate(samples, wanted_highest_sym, actual_highest_sym);
        samples2 = samples; //copy
        #ifdef BUILD_WITH_CUDA_SUPPORT
        samples3 = samples;
        #endif
        std::cout << "Total samples: " << samples.size() << "\n";
        std::cout << "Highest symbol: " << actual_highest_sym << "\n";
    }

    std::unique_ptr<cmodule_if> simple (new c_simple_mod());
    simple->run(samples,actual_highest_sym, output);
 
    std::cout <<  "\n";

    std::unique_ptr<cmodule_if> simple2 (new c_simple_4arith_mod());
    simple2->run(samples2,actual_highest_sym, output2);

    #ifdef BUILD_WITH_CUDA_SUPPORT
    std::cout <<  "\n";

    std::unique_ptr<cmodule_if> simple3 (new c_simple_cuda_mod());
    simple3->run(samples3,actual_highest_sym, output3);

    //for (auto p: output) {
    //    std::cout << p.key << "." << p.val << "\n";
    //}
    #endif
 

    return 0;
}
