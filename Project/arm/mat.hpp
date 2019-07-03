#include "util.hpp"

template<uint m, uint n, uint p>
unique_ptr<float[]> mul(float* a, float* b);

template<uint m, uint n>
void ternarize(float* a, float weight_pos, float weight_neg, float threshold);

template<uint m, uint n>
unique_ptr<float[]> transpose(float* a);

template<uint batch_size, uint frame_size>
void normalize(float* buf);
