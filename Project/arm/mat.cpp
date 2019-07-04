#include "mat.hpp"

#include <cassert>
#include <cmath>

#include <arm_neon.h>

template<uint m, uint n, uint p>
unique_ptr<float[]> mul(float* a, float* b)
{
	unique_ptr<float[]> c(new float[m * p]);
	for(uint i = 0; i < m; ++i)
	{
		for(uint j = 0; j < p; ++j)
		{
			float c_ij = 0;
			for(uint k = 0; k < n; ++k) c_ij += a[i * n + k] * b[k * p + j];
			c[i * p + j] = c_ij;
		}
	}
	return c;
}

template<uint rowA, uint colArowB, uint colB>
unique_ptr<float[]> mulJona(float* a, float* b)
{
	unique_ptr<float[]> c(new float[rowA * colB]);

	for(uint i = 0; i < rowA; ++i)
	{
		for(uint j = 0; j < colB; ++j)
		{
			float c_ij = 0;
			for(uint k = 0; k < colArowB; ++k)
			{
				c_ij += a[i * colArowB + k] * b[k * colB + j];
			}
			c[i * colB + j] = c_ij;
		}
	}
	return c;
}

template<uint m, uint n>
float max(float* a)
{
	float result = 0.0;
	for(float* ptr = a; ptr < a + (m * n); ++ptr)
	{
		if(fabs(*ptr) > result) result = fabs(*ptr);
	}
	return result;
}

template<uint m, uint n>
void ternarize(float* a, float weight_pos, float weight_neg, float threshold)
{
	float delta = 0.4 * max<m, n>(a);
	uint z_cntr = 0;

	for(float* ptr = a; ptr < a + (m * n); ++ptr)
	{
		if(*ptr >= delta)
		{
			*ptr = weight_pos;
		} else if(*ptr <= -delta)
		{
			*ptr = weight_neg;
		} else
		{
			*ptr = 0.0;
			++z_cntr;
		}
	}

	double sparsity = (double) z_cntr / (m * n);
	LOG(sparsity);
}

template<uint m, uint n>
unique_ptr<float[]> transpose(float* a)
{
	unique_ptr<float[]> t(new float[n * m]);
	for(uint i = 0; i < m; ++i)
	{
		for(uint j = 0; j < n; ++j) t[j * m + i] = a[i * n + j];
	}
	return t;
}

template<uint batch_size, uint frame_size>
void normalize(float* buf)
{
	for(float* ptr = buf; ptr < buf + batch_size * frame_size; ++ptr)
	{
		assert(*ptr >= 0 && *ptr <= 255);
		*ptr = *ptr / 255;
	}
}

template<uint num_neurons>
unique_ptr<float[]> compute_zeta(float* gamma, float* variance)
{
	unique_ptr<float[]> zeta(new float[num_neurons]);
	for(int i = 0; i < num_neurons; ++i) { zeta[i] = gamma[i] / sqrt(variance[i] + 1e-4); }
	return zeta;
}

template<uint num_neurons, uint batch_size>
uint ReLU(float* InputTensor, float threshold)
{
    float n = 255;
    uint zero_count = 0;
    for(int i = 0; i < batch_size * num_neurons; ++i)
    {
        if(InputTensor[i] < threshold)
        {
            InputTensor[i] = 0.0;
        } else if(InputTensor[i] > 1)
        {
            InputTensor[i] = 1.0;
        } else
        {
            InputTensor[i] = (round(InputTensor[i] * n) / n);
        }
            
        if(InputTensor[i] == 0) zero_count++;
    }
    return zero_count;
}

template<uint batch_size, uint num_units>
void Softmax(float* logits)
{
    float max, sum;

    for(int id = 0; id < batch_size; ++id)
    {
        max = 0.0;
        sum = 0.0;
        for(int i = 0; i < num_units; i++)
            if(max < logits[id * num_units + i]) max = logits[id * num_units + i];
        for(int i = 0; i < num_units; ++i)
        {
            logits[id * num_units + i] = exp(logits[id * num_units + i] - max);
            sum += logits[id * num_units + i];
        }
        for(int i = 0; i < num_units; ++i) { logits[id * num_units + i] /= sum; }
    }
}

template<uint m, uint n>
void batch_normalization(float* in, float* beta, float* gamma, float* mean, float* variance)
{
	for(int i = 0; i < n; ++i)
	{
		for(int j = 0; j < m; ++j)
			in[j * n + i] = ((in[j * n + i] - mean[j]) * gamma[j]) / sqrt(variance[j] + 1e-4) + beta[j];
	}
}

template<uint batch_size, uint num_neurons>
void batch_normalization_arm(float32_t *in, float32_t *beta, float32_t *mean, float32_t *zeta)
{
	float32x4_t mean4ps;
	float32x4_t beta4ps;
	float32x4_t zeta4ps;

	for(int i = 0; i < batch_size; ++i)
	{
		for(int j = 0; j < num_neurons; j += 4) {
			mean4ps = vld1q_f32(mean+j);
			beta4ps = vld1q_f32(beta+j);
			zeta4ps = vld1q_f32(zeta+j);

			float32x4_t input = vld1q_f32(in + (i * num_neurons + j));
			float32x4_t sub = vsubq_f32(input, mean4ps);
			float32x4_t result = vfmaq_f32(beta4ps, sub, zeta4ps);
			vst1q_f32(in + (i * num_neurons + j), result);
		}
	}
}

template<uint batch_size, uint num_neurons>
void batch_normalizationJona(float* in, float* beta, float* gamma, float* mean, float* variance)
{
	for(int i = 0; i < batch_size; ++i)
	{
		for(int j = 0; j < num_neurons; ++j)
			in[i * num_neurons + j] = ((in[i * num_neurons + j] - mean[j]) * gamma[j]) / sqrt(variance[j] + 1e-4) + beta[j];
	}
}

template<uint batch_size, uint num_units>
float get_accuracy(float* probs, int* labels)
{
	float max;
	int pred_class;
	int correct = 0;

	for(int id = 0; id < batch_size; ++id)
	{
		// Get class with highest probability
		max        = 0.0;
		pred_class = 0;
		for(int i = 0; i < num_units; i++)
		{
			if(max < probs[id * num_units + i])
			{
				max        = probs[id * num_units + i];
				pred_class = i;
			}
		}
		// Check against label
		if(pred_class == labels[id]) correct++;
	}
#ifdef PRINT_STATS
	std::cout << "[STATS] Correct = " << correct << "/" << batch_size << std::endl;
#endif

	return (float) correct / batch_size;
}

/* explicit template instantiations */
template unique_ptr<float[]> mul<batch_size, num_neurons, num_units>(float* a, float* b);
template unique_ptr<float[]> mulJona<batch_size, frame_size, num_neurons>(float* a, float* b);
template unique_ptr<float[]> mulJona<batch_size, num_neurons, num_units>(float* a, float* b);

template void ternarize<frame_size, num_neurons>(float* a, float weight_pos, float weight_neg, float threshold);
template void ternarize<num_neurons, num_neurons>(float* a, float weight_pos, float weight_neg, float threshold);

template unique_ptr<float[]> transpose<frame_size, num_neurons>(float* a);
template unique_ptr<float[]> transpose<num_neurons, num_neurons>(float* a);
template unique_ptr<float[]> transpose<batch_size, frame_size>(float* a);
template unique_ptr<float[]> transpose<num_neurons, batch_size>(float* a);

template void normalize<batch_size, frame_size>(float* buf);

template unique_ptr<float[]> compute_zeta<num_neurons>(float* gamma, float* variance);
															   
template uint ReLU<num_neurons, batch_size>(float* InputTensor, float threshold);
template void Softmax<batch_size, num_units>(float* logits);
template float get_accuracy<batch_size, num_units>(float* probs, int* labels);

template void batch_normalization<num_neurons, batch_size>(float* in, float* beta, float* gamma, float* mean,float* variance);

template void batch_normalizationJona<batch_size, num_neurons>(float* in, float* beta, float* gamma, float* mean,float* variance);

template void batch_normalization_arm<batch_size, num_neurons>(float* in, float* beta, float* mean, float* zeta);