#include <memory>
#include <array>
#include <stdexcept>

using std::unique_ptr;
using std::array;

struct sparse_list_t
{
	uint n;
	unique_ptr<uint[]> i;
};

struct sparse_list_tuple
{
	sparse_list_t pos, neg;
};

template<uint n, uint m>
unique_ptr<array<sparse_list_tuple, m>> createSparseList(float* weight_tensor, float weight_pos, float weight_neg)
{
	unique_ptr<array<sparse_list_tuple, m>> out;

	uint cntr_pos, cntr_neg;
	array<uint, m> buf_pos, buf_neg;

	for(uint row = 0; row < m; ++row)
	{
		cntr_pos = 0;
		cntr_neg = 0;
		for(uint col = 0; col < n; ++col)
		{
			switch(weight_tensor[row*n + col])
			{
				case weight_pos:
					buf_pos[cntr_pos++] = col;
					break;
				case weight_neg:
					buf_neg[cntr_neg++] = col;
					break;
				case 0:
					break;
				default:
					throw std::range_error("weight out of range");
			}
		}

		uint* i_pos = (uint*) malloc(cntr_pos * sizeof(uint));
		for(uint i = 0; i < cntr_pos; ++i) i_pos[i] = buf_pos[i];
		out[row].pos.n = cntr_pos;
		out[row].pos.i = i_pos;

		uint* i_neg = (uint*) malloc(cntr_neg * sizeof(uint));
		for(uint i = 0; i < cntr_neg; ++i) i_neg[i] = buf_neg[i];
		out[row].neg.n = cntr_neg;
		out[row].neg.i = i_neg;
	}

	return out;
}
