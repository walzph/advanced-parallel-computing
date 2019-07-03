#include "util.hpp"

template<uint m, uint n, uint p>
unique_ptr<float[]> mul(float* a, float* b);

template<uint m, uint n>
void ternarize(float* a, float weight_pos, float weight_neg, float threshold);

template<uint m, uint n>
unique_ptr<float[]> transpose(float* a);

template<uint batch_size, uint frame_size>
void normalize(float* buf);


template<uint num_neurons>
unique_ptr<float[]> compute_zeta(float* gamma, float* variance);

template<uint num_neurons, uint batch_size>
void BatchnormalizationCMOZeta(float* InputTensor, float* beta, float* mean, float* zeta);


template<uint num_neurons, uint batch_size>
uint ReLU(float* InputTensor, float threshold);

template<uint batch_size, uint num_units>
void Softmax(float* logits);