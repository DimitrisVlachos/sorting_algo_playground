#pragma once
#include "license_hdr.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <limits>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cassert>
#include <cstdio>

#define likely(x)    (x)/*   __builtin_expect((x),1)*/
#define unlikely(x)   (x)/*  __builtin_expect((x),0)*/
//#include "smmintrin.h" //4.1
//#include <immintrin.h>


template <typename key_t, typename val_t>
struct pair_t {
    key_t key;
    val_t val;

    inline void operator=(const pair_t& other) {
        key = other.key;
        val = other.val;
    }
} __attribute__ ((aligned ( sizeof(void*)))) ;
