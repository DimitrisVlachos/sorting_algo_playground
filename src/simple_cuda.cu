
#ifdef BUILD_WITH_CUDA_SUPPORT
#include "license_hdr.hpp"
#include "base_includes.hpp"
#include "generators.hpp"
#include "perf.hpp"
#include "module.hpp"
#include "simple_cuda.hpp"
#include "cuda_buffer.hpp"


template <typename probability_t>
__global__ void zero_frequencies(probability_t* counts, const size_t size) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        counts[i] = 0;
}

template <typename base_t, typename probability_t>
__global__ void count_frequencies(const base_t* in, const size_t size, probability_t* count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        ++count[ in[i] ];
}

template <typename key_t, typename val_t > 
__global__ void bubble_sort(pair_t<key_t, val_t>* arr, const size_t sz)  {
    const size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= sz)
        return;
    //bool swapped = true;
     { 
        const size_t range = (sz-n-1);
        //swapped=false;
        for (size_t j = 0; likely(j < range); ++j)  { 
            if (likely(arr[j].val > arr[j+1].val)) { 
                const key_t t = arr[j].val;  
                arr[j].val = arr[j + 1].val;
                arr[j + 1].val = t;
                //swapped = true; 
            } 
     }
   } 
}
 
template <typename base_t, typename probability_t>
__global__ void id_frquencies(const probability_t* probs, const size_t size, pair_t<base_t, probability_t>* pairs) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        pairs[i].key = i;
        pairs[i].val = probs[i];
    }
}

    c_simple_cuda_mod::c_simple_cuda_mod() {

    }
    c_simple_cuda_mod::~c_simple_cuda_mod() {
        
    }
    bool c_simple_cuda_mod::run(std::vector<uint32_t>& samples, const size_t highest_symbol, std::vector<pair_t<uint32_t, uint32_t>>& output) {
        c_perf("c_simple_cuda_mod total time");

        cuda_buffer_c<uint32_t> probs(highest_symbol + 1);
        cuda_buffer_c<uint32_t> psamples(samples.size(), &samples[0]);
        cuda_buffer_c<pair_t<uint32_t, uint32_t>> poutput;
        
        poutput.alloc_pod(highest_symbol + 1, &output[0]);

        constexpr size_t block_sz = CUDA_BLOCKS;
        
        {
            c_perf("Init+Zero Freqs");
            const size_t block_num = ((highest_symbol+1) + block_sz - 1) / block_sz;
            zero_frequencies<<< block_num, block_sz>>>(probs.get_dev_buf(),(size_t)(highest_symbol+1));
        }

        {
            c_perf("Counting Freqs");
            const size_t block_num = (samples.size() + block_sz - 1) / block_sz;
            psamples.host2dev();
            psamples.sync();
            count_frequencies<<<block_num, block_sz>>>(&psamples.get_dev_buf()[0], samples.size(), &probs.get_dev_buf()[0]);
        }

        {
            c_perf("ID Freqs");
            const size_t block_num = ((highest_symbol+1) + block_sz - 1) / block_sz;
            id_frquencies<<<block_num, block_sz>>>(&probs.get_dev_buf()[0], highest_symbol + 1,&poutput.get_dev_buf()[0]);
        }
        
        {
            c_perf("Sorting");
            const size_t block_num = ((highest_symbol+1) + block_sz - 1) / block_sz;
            bubble_sort<<<block_num, block_sz>>>(&poutput.get_dev_buf()[0], highest_symbol + 1);
        }

        {
            c_perf("Flushing");
            poutput.dev2host();
            poutput.sync();
        }
        return true;
    }
    bool c_simple_cuda_mod::term() {
        return true;
    }

#endif
