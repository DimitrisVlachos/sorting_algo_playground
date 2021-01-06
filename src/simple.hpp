#pragma once
#include "module.hpp"

class c_simple_mod : public cmodule_if {
    private:
    public:
    ~c_simple_mod();
    c_simple_mod();
    bool run(std::vector<uint32_t>& samples, const size_t highest_symbol, std::vector<pair_t<uint32_t, uint32_t>>& output) ;
    bool term() ;
};

