
#include "base_includes.hpp"
#include "generators.hpp"
#include "perf.hpp"
#include "simple_4_arith.hpp"
 
template <typename key_t, typename val_t > 
static void bubble_sort(std::vector<pair_t<key_t, val_t>>& arr)  { //Try to use 4 arithmetic ports
    bool swapped;
    const size_t n = arr.size();

    for (size_t i = 0; likely(i < n-1); ++i )  { 
        swapped = false;
        const size_t range = (n-i-1);
        size_t j = 0;

        auto swp = [](auto& a, auto& b) {auto tmp = a; a = b; b = tmp;};

        for (;  likely( j+4  < range); j += 4 - 1)  { 
            auto& fetch0 = arr[j];
            auto& fetch1 = arr[j + 1];
            auto& fetch2 = arr[j + 2];
            auto& fetch3 = arr[j + 3];
  
            if (likely(fetch0.val > fetch1.val)) { swp(fetch0,fetch1); swapped = true; } 
            if (likely(fetch1.val > fetch2.val)) { swp(fetch1,fetch2); swapped = true; }
            if (likely(fetch2.val > fetch3.val)) { swp(fetch2,fetch3); swapped = true; } 
        }
 
        //Remainder
        for (; likely(j < range); ++j)  { 
            if (likely(arr[j].val > arr[j+1].val)) { 
                swp(arr[j],arr[j+1]);
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
    size_t i = 0;

    //Use 4 arith ports
    for ( ; i+4 < in.size();i += 4) {
        auto a = in[i];
        auto b = in[i+1];
        auto c = in[i+2];
        auto d = in[i+3];

        ++res[ a ];
        ++res[ b ];
        ++res[ c ];
        ++res[ d ];
    }

    //Remainder
    for ( ;i < in.size();++i) {
        ++res[ in[i] ];
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


    c_simple_4arith_mod::~c_simple_4arith_mod() {

    }

    c_simple_4arith_mod::c_simple_4arith_mod(){
        
    }

    bool c_simple_4arith_mod::run(std::vector<uint32_t>& samples, const size_t highest_symbol, std::vector<pair_t<uint32_t, uint32_t>>& output) {
        c_perf("c_simple_4arith_mod total time");

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
    bool c_simple_4arith_mod::term() {
        return true;
    }

