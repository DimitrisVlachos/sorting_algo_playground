#pragma once
#include "license_hdr.hpp"

#include "base_includes.hpp"

class c_perf {
    private:
    std::chrono::high_resolution_clock::time_point t1;
    std::string m_desc;
    public:

    c_perf(const std::string& desc) : m_desc(desc) {
        std::cout << "Starting perf: " << desc << "\n";
        t1 = std::chrono::high_resolution_clock::now();
    }

    ~c_perf() {
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::duration<double,std::nano> >(t2 - t1);
        auto ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1);
        //auto s = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1);
        std::cout << "Finished perf: " << m_desc << ". Took [" << std::fixed << std::setprecision(5) << (double)ns.count() << " ns]" << " OR [" << (double)ms.count() << " ms]"  <<".\n";
    }
};
 