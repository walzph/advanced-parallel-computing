#pragma once

#include <iostream>
#include <memory>
#include <vector>

typedef unsigned int uint;
#ifdef DEBUG
#define LOG(var) std::cout << #var "=" << var << "\n"
#else
#define LOG(var)
#endif

using std::unique_ptr;
using std::vector;

const uint batch_size  = BATCH_SIZE;
const uint frame_size  = 28 * 28;
const uint num_neurons = 1024;
const uint num_units   = 10;
