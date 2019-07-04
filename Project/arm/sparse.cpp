#include "sparse.hpp"

#include <algorithm>
#include <stdexcept>

using std::copy;

template<uint m, uint n>
unique_ptr<sparse_list_tuple[]> createSparseList(const float* weight_tensor, const float weight_pos,
                                                 const float weight_neg)
{
	unique_ptr<sparse_list_tuple[]> out(new sparse_list_tuple[m]);

	uint cntr_pos, cntr_neg;
	unique_ptr<uint[]> buf_pos(new uint[n]), buf_neg(new uint[n]);

	for(uint row = 0; row < m; ++row)
	{
		cntr_pos = 0;
		cntr_neg = 0;
		for(uint col = 0; col < n; ++col)
		{
			float weight = weight_tensor[row * n + col];
			if(weight == weight_pos)
			{
				buf_pos[cntr_pos++] = col;
			} else if(weight == weight_neg)
			{
				buf_neg[cntr_neg++] = col;
			} else if(weight != 0)
			{
				throw std::range_error("weight out of range");
			}
		}

		out[row].pos.n = cntr_pos;
		out[row].pos.i = unique_ptr<uint[]>(new uint[cntr_pos]);
		copy(buf_pos.get(), buf_pos.get() + cntr_pos, out[row].pos.i.get());

		out[row].neg.n = cntr_neg;
		out[row].neg.i = unique_ptr<uint[]>(new uint[cntr_neg]);
		copy(buf_neg.get(), buf_neg.get() + cntr_neg, out[row].neg.i.get());
	}

	return out;
}

template<uint m, uint p>
unique_ptr<float[]> sparseMatrixMultiply(const float* input, const sparse_list_tuple* sparse_lists,
                                         const float weight_pos, const float weight_neg)
{
	unique_ptr<float[]> out(new float[m * p]);

#pragma omp parallel for
	for(uint i = 0; i < m; ++i)
	{
		for(uint j = 0; j < p; ++j)
		{
			float pos = 0, neg = 0;

			uint pos_n = sparse_lists[i].pos.n;
			uint neg_n = sparse_lists[i].neg.n;

			const sparse_list_tuple& indices = sparse_lists[i];

			// printf("pos: ");
			for(uint k = 0; k < pos_n; ++k)
			{
				uint idx = indices.pos.i[k];
				// printf("%d ", idx);
				pos += input[idx * p + j];
			}
			// printf("\n");

			// printf("neg: ");
			for(uint k = 0; k < neg_n; ++k)
			{
				uint idx = indices.neg.i[k];
				// printf("%d ", idx);
				neg += input[idx * p + j];
			}
			// printf("\n");

			out[i * p + j] = weight_pos * pos + weight_neg * neg;
		}
	}

	return out;
}

/* explicit template instantiations */
template unique_ptr<sparse_list_tuple[]>
createSparseList<num_neurons, frame_size>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<sparse_list_tuple[]>
createSparseList<num_neurons, num_neurons>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<sparse_list_tuple[]>
createSparseList<num_units, num_neurons>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<float[]> sparseMatrixMultiply<num_neurons, batch_size>(const float* input,
                                                                           const sparse_list_tuple* sparse_lists,
                                                                           const float weight_pos,
                                                                           const float weight_neg);

template unique_ptr<float[]> sparseMatrixMultiply<num_units, batch_size>(const float* input,
                                                                         const sparse_list_tuple* sparse_lists,
                                                                         const float weight_pos,
                                                                         const float weight_neg);
