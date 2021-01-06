#pragma once
#include "license_hdr.hpp"

#ifdef BUILD_WITH_CUDA_SUPPORT
#include "base_includes.hpp"

template <typename base_t>
class cuda_buffer_c {
private:
    base_t* host_buf;
    base_t* dev_buf;
    size_t data_count;
    bool shared_host_ptr;

    inline void cleanup() {
        if (!shared_host_ptr)
            delete[] host_buf;

        if (dev_buf)
            cudaFree(dev_buf);

        dev_buf = host_buf = nullptr;
        data_count = 0;
        shared_host_ptr = false;
    }

public:
    cuda_buffer_c() : host_buf(nullptr), dev_buf(nullptr), shared_host_ptr(false) {}
    ~cuda_buffer_c() { cleanup(); }
    cuda_buffer_c(const size_t count,const bool reset_host_mem = false, const base_t default_host_val = (base_t)0) : host_buf(nullptr), dev_buf(nullptr), shared_host_ptr(false) {
        this->alloc(count,nullptr,reset_host_mem,default_host_val);
    }

    cuda_buffer_c(base_t* host_ptr, const size_t count,const bool reset_host_mem = false, const base_t default_host_val = (base_t)0) : host_buf(nullptr), dev_buf(nullptr), shared_host_ptr(false) {
        this->alloc(count,host_ptr,reset_host_mem,default_host_val);
    }

    inline base_t* get_host_buf() { return host_buf; }
    inline base_t* get_dev_buf() { return dev_buf; }

    inline void alloc(const size_t count, base_t* host_ptr = nullptr, const bool reset_host_mem = false, const base_t default_host_val = 0) {
        this->cleanup();
        this->data_count = count;
        this->shared_host_ptr = host_ptr != nullptr;
        cudaMalloc(&dev_buf, count*sizeof(base_t)); 
        assert(dev_buf != nullptr);

        //We use malloc to avoid new[]::ctor() call
        if (!shared_host_ptr)
            host_buf = static_cast<base_t*>(malloc(count*sizeof(base_t)));
        else
            host_buf = host_ptr;

        assert(host_buf != nullptr);
        if (reset_host_mem) {
            for (size_t i = 0;i < count;++i)
                host_buf[i] = default_host_val;
        }
    }
 
     inline void alloc_pod(const size_t count, base_t* host_ptr = nullptr, const bool reset_host_mem = false) {
        this->cleanup();
        this->data_count = count;
        this->shared_host_ptr = host_ptr != nullptr;
        cudaMalloc(&dev_buf, count*sizeof(base_t)); 
        assert(dev_buf != nullptr);

        //We use malloc to avoid new[]::ctor() call
        if (!shared_host_ptr)
            host_buf = static_cast<base_t*>(malloc(count*sizeof(base_t)));
        else
            host_buf = host_ptr;

        assert(host_buf != nullptr);
        if (reset_host_mem) {
            memset((void*)host_buf,0,count*sizeof(base_t));
        }
    }


    inline void dev2host() {
        assert(host_buf != nullptr);
        assert(dev_buf != nullptr);
        cudaMemcpy(host_buf, dev_buf, data_count*sizeof(base_t), cudaMemcpyDeviceToHost );
    }

    inline void host2dev() {
        assert(host_buf != nullptr);
        assert(dev_buf != nullptr);
        cudaMemcpy(dev_buf, host_buf, data_count*sizeof(base_t), cudaMemcpyHostToDevice);
    }

    inline void sync() {
        cudaDeviceSynchronize();
    }
};
#endif