#include "base_includes.hpp"
#include "generators.hpp"
#include "perf.hpp"
#include "module.hpp"
#include "simple.hpp"

template <typename key_t, typename val_t > 
static void bubble_sort(std::vector<pair_t<key_t, val_t>>& arr)  {
    const size_t n = arr.size();
    if ((ssize_t)n < 1)
        return;
    bool swapped = true;
    for (size_t i = 0; likely(i < n-1); ++i )  { 
        const size_t range = (n-i-1);
        swapped=false;
        for (size_t j = 0; likely(j < range); ++j)  { 
            if (likely(arr[j].val > arr[j+1].val)) { 
                const key_t t = arr[j].val;  
                arr[j].val = arr[j + 1].val;
                arr[j + 1].val = t;
                swapped = true; 
            } 
     }
     if (unlikely(!swapped)) 
        return;
   } 
}

template <typename base_t, typename probability_t>
static void count_frequencies(const std::vector<base_t>& in, std::vector<probability_t>& res, const size_t highest_symbol) {
    res.clear();
    res.resize(highest_symbol + 1);

    for (const auto p : in) {
        ++res[p];
    }
}

template <typename base_t, typename probability_t>
static void sort_frequencies(const std::vector<probability_t>& probs, std::vector<pair_t<base_t, probability_t>>& res) {
    res.clear();
    res.resize(probs.size());

    for (size_t i = 0;i < probs.size();++i) {
        res[i].key = (base_t)i;
        res[i].val = probs[i];
    }

    bubble_sort<base_t, probability_t>(res);
}

    c_simple_mod::c_simple_mod() {

    }
    c_simple_mod::~c_simple_mod() {
        
    }
    bool c_simple_mod::run(std::vector<uint32_t>& samples, const size_t highest_symbol, std::vector<pair_t<uint32_t, uint32_t>>& output) {
        c_perf("c_simple_mod total time");

        std::vector<uint32_t> probs;

        {
            c_perf("Counting Freqs");
            count_frequencies(samples, probs, highest_symbol);
        }

        {
            c_perf("Sorting Freqs");
            sort_frequencies(probs, output);
        }
        return true;
    }
    bool c_simple_mod::term() {
        return true;
    }

