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

template<uint row, uint col>
unique_ptr<sparse_list_tuple[]> createSparseListJona(const float* weight_tensor, const float weight_pos,
                                                 const float weight_neg)
{
	unique_ptr<sparse_list_tuple[]> out(new sparse_list_tuple[col]);

	uint cntr_pos, cntr_neg;
	unique_ptr<uint[]> buf_pos(new uint[row]), buf_neg(new uint[row]);

	for(uint colIndex = 0; colIndex < col; ++colIndex)
	{
		cntr_pos = 0;
		cntr_neg = 0;
		for(uint rowIndex = 0; rowIndex < row; ++rowIndex)
		{
			float weight = weight_tensor[rowIndex * col + colIndex];
			if(weight == weight_pos)
			{
				buf_pos[cntr_pos++] = rowIndex;
			} else if(weight == weight_neg)
			{
				buf_neg[cntr_neg++] = rowIndex;
			} else if(weight != 0)
			{
				throw std::range_error("weight out of range");
			}
		}

		out[colIndex].pos.n = cntr_pos;
		out[colIndex].pos.i = unique_ptr<uint[]>(new uint[cntr_pos]);
		copy(buf_pos.get(), buf_pos.get() + cntr_pos, out[colIndex].pos.i.get());

		out[colIndex].neg.n = cntr_neg;
		out[colIndex].neg.i = unique_ptr<uint[]>(new uint[cntr_neg]);
		copy(buf_neg.get(), buf_neg.get() + cntr_neg, out[colIndex].neg.i.get());
	}

	return out;
	// unique_ptr<sparse_list_tuple[]> out(new sparse_list_tuple[row]);

	// uint cntr_pos, cntr_neg;
	// unique_ptr<uint[]> buf_pos(new uint[col]), buf_neg(new uint[col]);

	// for(uint rowIndex = 0; rowIndex < row; ++rowIndex)
	// {
	// 	cntr_pos = 0;
	// 	cntr_neg = 0;
	// 	for(uint colIndex = 0; colIndex < col; ++colIndex)
	// 	{
	// 		float weight = weight_tensor[rowIndex * col + colIndex];
	// 		if(weight == weight_pos)
	// 		{
	// 			buf_pos[cntr_pos++] = colIndex;
	// 		} else if(weight == weight_neg)
	// 		{
	// 			buf_neg[cntr_neg++] = colIndex;
	// 		} else if(weight != 0)
	// 		{
	// 			throw std::range_error("weight out of range");
	// 		}
	// 	}

	// 	out[rowIndex].pos.n = cntr_pos;
	// 	out[rowIndex].pos.i = unique_ptr<uint[]>(new uint[cntr_pos]);
	// 	copy(buf_pos.get(), buf_pos.get() + cntr_pos, out[rowIndex].pos.i.get());

	// 	out[rowIndex].neg.n = cntr_neg;
	// 	out[rowIndex].neg.i = unique_ptr<uint[]>(new uint[cntr_neg]);
	// 	copy(buf_neg.get(), buf_neg.get() + cntr_neg, out[rowIndex].neg.i.get());
	// }

	// return out;
}

template<uint m, uint p>
unique_ptr<float[]> sparseMatrixMultiply(const float* input, const sparse_list_tuple* sparse_lists,
                                         const float weight_pos, const float weight_neg)
{
	unique_ptr<float[]> out(new float[m * p]);

	for(uint i = 0; i < m; ++i)
	{
		for(uint j = 0; j < p; ++j)
		{
			float pos = 0, neg = 0;

			uint pos_n = sparse_lists[i].pos.n;
			uint neg_n = sparse_lists[i].neg.n;

			const sparse_list_tuple& indices = sparse_lists[i];

			for(uint k = 0; k < pos_n; ++k)
			{
				uint idx = indices.pos.i[k];
				pos += input[i * p + idx];
			}

			for(uint k = 0; k < neg_n; ++k)
			{
				uint idx = indices.neg.i[k];
				neg += input[i * p + idx];
			}

			out[i * p + j] = weight_pos * pos + weight_neg * neg;
		}
	}

	return out;
}


template<uint rowA, uint colArowB, uint colB>
unique_ptr<float[]> sparseMatrixMultiplyJona(const float* input, const sparse_list_tuple* sparse_lists,
                                         const float weight_pos, const float weight_neg)
{
	unique_ptr<float[]> out(new float[rowA * colB]);

	#pragma omp parallel for
	for(uint i = 0; i < rowA; ++i)
	{		
		for(uint j = 0; j < colB; ++j)
		{
			float pos = 0, neg = 0;

			uint pos_n = sparse_lists[j].pos.n;
			uint neg_n = sparse_lists[j].neg.n;

			const sparse_list_tuple& indices = sparse_lists[j];

			for(uint k = 0; k < pos_n; ++k)
			{
				uint idx = indices.pos.i[k];
				pos += input[i * colArowB + idx];
			}

			for(uint k = 0; k < neg_n; ++k)
			{
				uint idx = indices.neg.i[k];
				neg += input[i * colArowB + idx];
			}
			
			out[i * colB + j] = weight_pos * pos + weight_neg * neg;
		}
	}

	return out;
}



/* explicit template instantiations */
// template unique_ptr<sparse_list_tuple[]>
// createSparseList<num_neurons, frame_size>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<sparse_list_tuple[]>
createSparseList<num_neurons, num_neurons>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<sparse_list_tuple[]>
createSparseListJona<frame_size, num_neurons>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<sparse_list_tuple[]>
createSparseListJona<num_neurons, num_units>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<sparse_list_tuple[]>
createSparseListJona<num_neurons, num_neurons>(const float* weight_tensor, const float weight_pos, const float weight_neg);

template unique_ptr<float[]> sparseMatrixMultiply<num_neurons, batch_size>(const float* input,
                                                                           const sparse_list_tuple* sparse_lists,
                                                                           const float weight_pos,
                                                                           const float weight_neg);

template unique_ptr<float[]> sparseMatrixMultiply<num_units, batch_size>(const float* input,
                                                                           const sparse_list_tuple* sparse_lists,
                                                                           const float weight_pos,
                                                                           const float weight_neg); 

template unique_ptr<float[]> sparseMatrixMultiply<batch_size, num_neurons>(const float* input,
                                                                           const sparse_list_tuple* sparse_lists,
                                                                           const float weight_pos,
                                                                           const float weight_neg); 																		   

template unique_ptr<float[]> sparseMatrixMultiplyJona<batch_size, frame_size, num_neurons>(const float* input,
                                                                           const sparse_list_tuple* sparse_lists,
                                                                           const float weight_pos,
                                                                           const float weight_neg);		

template unique_ptr<float[]> sparseMatrixMultiplyJona<batch_size, num_neurons, num_units>(const float* input,
                                                                           const sparse_list_tuple* sparse_lists,
                                                                           const float weight_pos,
                                                                           const float weight_neg);		

template unique_ptr<float[]> sparseMatrixMultiplyJona<batch_size, num_neurons, num_neurons>(const float* input,
                                                                           const sparse_list_tuple* sparse_lists,
                                                                           const float weight_pos,
                                                                           const float weight_neg);																			   																   																   
