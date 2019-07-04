#pragma once

#include <iostream>
#include <memory>
#include <vector>

typedef unsigned int uint;
#define LOG(var) std::cout << #var "=" << var << "\n";
//#define TEST

using std::unique_ptr;
using std::vector;

#ifdef BATCH_SIZE
const uint batch_size = BATCH_SIZE;
#else
const uint batch_size = 64;
#endif
const uint frame_size  = 28 * 28;
const uint num_neurons = 1024;
const uint num_units   = 10;
