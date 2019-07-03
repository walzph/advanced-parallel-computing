#pragma once

#include <iostream>
#include <memory>
#include <vector>

typedef unsigned int uint;
#define LOG(var) std::cout << #var "=" << var << "\n";

using std::unique_ptr;
using std::vector;

const uint batch_size  = 1; // 64;
const uint frame_size  = 28 * 28;
const uint num_neurons = 1024;
const uint num_units   = 10;
