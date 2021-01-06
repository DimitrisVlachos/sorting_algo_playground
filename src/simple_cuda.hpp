#pragma once
#include "license_hdr.hpp"

#ifdef BUILD_WITH_CUDA_SUPPORT
#include "module.hpp"
class c_simple_cuda_mod : public cmodule_if {
    private:
    public:
    ~c_simple_cuda_mod();
    c_simple_cuda_mod();
    bool run(std::vector<uint32_t>& samples, const size_t highest_symbol, std::vector<pair_t<uint32_t, uint32_t>>& output) ;
    bool term() ;
};
#endif