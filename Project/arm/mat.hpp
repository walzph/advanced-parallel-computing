#include "util.hpp"
#include <arm_neon.h>

template<uint m, uint n, uint p>
unique_ptr<float[]> mul(float* a, float* b);

template<uint rowA, uint colArowB, uint colB>
unique_ptr<float[]> mulJona(float* a, float* b);

template<uint m, uint n>
void ternarize(float* a, float weight_pos, float weight_neg, float threshold);

template<uint m, uint n>
unique_ptr<float[]> transpose(float* a);

template<uint batch_size, uint frame_size>
void normalize(float* buf);

template<uint batch_size, uint num_neurons>
void batch_normalizationJona(float* in, float* beta, float* gamma, float* mean, float* variance);

template<uint num_neurons>
unique_ptr<float[]> compute_zeta(float* gamma, float* variance);

template<uint m, uint n>
void batch_normalization(float* in, float* beta, float* gamma, float* mean, float* variance);

template<uint batch_size, uint num_neurons>
void batch_normalization_arm(float32_t *in, float32_t *beta, float32_t *mean, float32_t *zeta);

template<uint num_neurons, uint batch_size>
uint ReLU(float* InputTensor, float threshold);

template<uint batch_size, uint num_units>
void Softmax(float* logits);

template<uint batch_size, uint num_units>
float get_accuracy(float* probs, int* labels);
