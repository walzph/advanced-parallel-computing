#include "util.hpp"

struct sparse_list
{
	uint n;
	unique_ptr<uint[]> i;
};

struct sparse_list_tuple
{
	sparse_list pos, neg;
};

template<uint m, uint n>
unique_ptr<sparse_list_tuple[]> createSparseList(const float* weight_tensor, const float weight_pos,
                                                 const float weight_neg);

template<uint rows, uint cols>
unique_ptr<sparse_list_tuple[]> createSparseListJona(const float* weight_tensor, const float weight_pos,
                                                 const float weight_neg);

template<uint m, uint p>
unique_ptr<float[]> sparseMatrixMultiply(const float* input, const sparse_list_tuple* sparse_lists,
                                         const float weight_pos, const float weight_neg);

template<uint rowA, uint colArowB, uint colB>
unique_ptr<float[]> sparseMatrixMultiplyJona(const float* input, const sparse_list_tuple* sparse_lists,
                                         const float weight_pos, const float weight_neg);
