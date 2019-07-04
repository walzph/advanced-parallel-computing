#include "sparse.hpp"

#include <algorithm>
#include <stdexcept>

using std::copy;

template<uint m, uint n>
unique_ptr<sparse_list_tuple[]> createSparseList(const float* weight_tensor, const float weight_pos,
                                                 const float weight_neg)
{
	unique_ptr<sparse_list_tuple[]> out(new sparse_list_tuple[n]);

	uint cntr_pos, cntr_neg;
	unique_ptr<uint[]> buf_pos(new uint[m]), buf_neg(new uint[m]);

	for(uint col = 0; col < n; ++col)
	{
		cntr_pos = 0;
		cntr_neg = 0;
		for(uint row = 0; row < m; ++row)
		{
			float weight = weight_tensor[row * n + col];
			if(weight == weight_pos)
			{
				buf_pos[cntr_pos++] = row;
			} else if(weight == weight_neg)
			{
				buf_neg[cntr_neg++] = row;
			} else if(weight != 0)
			{
				throw std::range_error("weight out of range");
			}
		}

		out[col].pos.n = cntr_pos;
		out[col].pos.i = unique_ptr<uint[]>(new uint[cntr_pos]);
		copy(buf_pos.get(), buf_pos.get() + cntr_pos, out[col].pos.i.get());

		out[col].neg.n = cntr_neg;
		out[col].neg.i = unique_ptr<uint[]>(new uint[cntr_neg]);
		copy(buf_neg.get(), buf_neg.get() + cntr_neg, out[col].neg.i.get());
	}

	return out;
}

template<uint m, uint n, uint p>
unique_ptr<float[]> sparseMatrixMultiply(const float* input, const sparse_list_tuple* sparse_lists,
                                         const float weight_pos, const float weight_neg)
{
	unique_ptr<float[]> out(new float[m * p]);

#pragma omp parallel for collapse(2)
	for(uint i = 0; i < m; ++i)
	{
		for(uint j = 0; j < p; ++j)
		{
			float pos = 0, neg = 0;

			const sparse_list_tuple& indices = sparse_lists[j];

			uint pos_n = indices.pos.n;
			uint neg_n = indices.neg.n;

#pragma omp simd
			for(uint k = 0; k < pos_n; ++k)
			{
				uint idx = indices.pos.i[k];
				pos += input[i * n + idx];
			}

#pragma omp simd
			for(uint k = 0; k < neg_n; ++k)
			{
				uint idx = indices.neg.i[k];
				neg += input[i * n + idx];
			}

			out[i * p + j] = weight_pos * pos + weight_neg * neg;
		}
	}

	return out;
}

/* explicit template instantiations */
template unique_ptr<sparse_list_tuple[]>
createSparseList<frame_size, num_neurons>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<sparse_list_tuple[]>
createSparseList<num_neurons, num_neurons>(const float* weight_tensor, const float weight_pos, const float weight_neg);

// template unique_ptr<sparse_list_tuple[]>
// createSparseList<num_units, num_neurons>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<float[]>
sparseMatrixMultiply<batch_size, frame_size, num_neurons>(const float* input, const sparse_list_tuple* sparse_lists,
                                                          const float weight_pos, const float weight_neg);

template unique_ptr<float[]>
sparseMatrixMultiply<batch_size, num_neurons, num_neurons>(const float* input, const sparse_list_tuple* sparse_lists,
                                                           const float weight_pos, const float weight_neg);

template unique_ptr<float[]>
sparseMatrixMultiply<batch_size, num_neurons, num_units>(const float* input, const sparse_list_tuple* sparse_lists,
                                                         const float weight_pos, const float weight_neg);
