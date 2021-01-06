#pragma once
#include "license_hdr.hpp"

#include "base_includes.hpp"


class cmodule_if {
    private:
    public:
    virtual ~cmodule_if() {}
    virtual bool run(std::vector<uint32_t>& samples, const size_t highest_symbol, std::vector<pair_t<uint32_t, uint32_t>>& output ) { return false; }
    virtual bool term() { return false; }
};


