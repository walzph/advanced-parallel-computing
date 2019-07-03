#include "mat.hpp"

#include <cassert>
#include <cmath>

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

/* explicit template instantiations */
template unique_ptr<float[]> mul<batch_size, num_neurons, num_units>(float* a, float* b);

template void ternarize<frame_size, num_neurons>(float* a, float weight_pos, float weight_neg, float threshold);
template void ternarize<num_neurons, num_neurons>(float* a, float weight_pos, float weight_neg, float threshold);

template unique_ptr<float[]> transpose<frame_size, num_neurons>(float* a);
template unique_ptr<float[]> transpose<num_neurons, num_neurons>(float* a);
template unique_ptr<float[]> transpose<frame_size, batch_size>(float* a);
template unique_ptr<float[]> transpose<num_neurons, batch_size>(float* a);

template void normalize<batch_size, frame_size>(float* buf);
