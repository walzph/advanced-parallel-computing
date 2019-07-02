#include "mat.hpp"

#include <cmath>

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

/* explicit template instantiations */
template void ternarize<frame_size, num_neurons>(float* a, float weight_pos, float weight_neg, float threshold);
template void ternarize<num_neurons, num_neurons>(float* a, float weight_pos, float weight_neg, float threshold);

template unique_ptr<float[]> transpose<frame_size, num_neurons>(float* a);
template unique_ptr<float[]> transpose<num_neurons, num_neurons>(float* a);
template unique_ptr<float[]> transpose<batch_size, frame_size>(float* a);
template unique_ptr<float[]> transpose<num_neurons, batch_size>(float* a);
