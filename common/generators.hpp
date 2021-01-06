#pragma once
#include "license_hdr.hpp"

#include "base_includes.hpp"

template <typename base_t>
class c_data_generator_if {
    public:
    virtual bool generate(std::vector<base_t>& output, const size_t highest_sym, size_t& actual_highest_sym) = 0;
};

template <typename base_t>
class c_data_generator_random_impl : public c_data_generator_if<base_t> {
    bool generate(std::vector<base_t>& output, const size_t highest_sym, size_t& actual_highest_sym) {

        const size_t n = 10 * 1000000;
        const size_t m = highest_sym;
        actual_highest_sym = highest_sym;
        output.clear();
        output.resize(n);
        
        for (size_t i = 0;i < n; ++i)
            output[i] = rand()%m;

        return true;
    }
};

template <typename base_t>
class c_data_generator_debug_impl : public c_data_generator_if<base_t> {
    bool generate(std::vector<base_t>& output, const size_t highest_sym, size_t& actual_highest_sym) {
        output.clear();
        output.resize(highest_sym);
        actual_highest_sym = highest_sym;
        for (size_t i = 0;i < highest_sym; ++i)
            output[i] = i;

        return true;
    }
};


