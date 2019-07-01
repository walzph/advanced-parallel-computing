#include <algorithm>
#include <cmath>
#include <iomanip>

#include <cnpy.h>
#include <mnist/mnist_reader.hpp>
#include <mnist/mnist_utils.hpp>

#include "mat.hpp"
#include "sparse.hpp"
#include "util.hpp"

using std::copy;

template<uint batch_size>
unique_ptr<float[]> init_input(const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>& data, uint id);

template<uint batch_size, uint frame_size>
void normalize(float* buf);

template<uint n, uint m>
void print_frame(float* frame)
{
	for(uint i = 0; i < n; ++i)
	{
		for(uint j = 0; j < m; ++j) std::cout << std::setw(4) << frame[i * m + j] << ' ';
		std::cout << '\n';
	}
}

int main(int argc, char* argv[])
{
	mnist::MNIST_dataset<vector, vector<float>, uint8_t> dataset =
	    mnist::read_dataset<vector, vector, float, uint8_t>("./mnist/");

	cnpy::npz_t parameters_npz = cnpy::npz_load("trained_models/mnist-t40-final.npz");

	assert(frame_size == parameters_npz["fc0/W:0"].shape[0]);
	assert(num_neurons == parameters_npz["fc0/W:0"].shape[1]);

	float* weight_tensor_0 = parameters_npz["fc0/W:0"].data<float>();

	float weight_pos_0 = *parameters_npz["fc0/Wp:0"].data<float>();
	float weight_neg_0 = -(*parameters_npz["fc0/Wn:0"].data<float>());

	assert(fabs(weight_pos_0 - 0.7502421) < 0.00001);
	assert(fabs(weight_neg_0 - -1.2224674) < 0.00001);

	ternarize<frame_size, num_neurons>(weight_tensor_0, weight_pos_0, weight_neg_0, 0.1f);
	unique_ptr<float[]> weight_tensor_0_t = transpose<frame_size, num_neurons>(weight_tensor_0);

	unique_ptr<sparse_list_tuple[]> sparse_lists_0 =
	    createSparseList<num_neurons, frame_size>(weight_tensor_0_t.get(), weight_pos_0, weight_neg_0);

	assert(num_neurons == parameters_npz["fc0/b:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn0/beta:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn0/gamma:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn0/mean/EMA:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn0/variance/EMA:0"].shape[0]);

	float* bias_0     = parameters_npz["fc0/b:0"].data<float>();
	float* beta_0     = parameters_npz["bn0/beta:0"].data<float>();
	float* gamma_0    = parameters_npz["bn0/gamma:0"].data<float>();
	float* mean_0     = parameters_npz["bn0/mean/EMA:0"].data<float>();
	float* variance_0 = parameters_npz["bn0/variance/EMA:0"].data<float>();


	assert(num_neurons == parameters_npz["fc1/W:0"].shape[0]);
	assert(num_neurons == parameters_npz["fc1/W:0"].shape[1]);

	float* weight_tensor_1 = parameters_npz["fc1/W:0"].data<float>();

	float weight_pos_1 = *parameters_npz["fc1/Wp:0"].data<float>();
	float weight_neg_1 = -(*parameters_npz["fc1/Wn:0"].data<float>());

	assert(fabs(weight_pos_1 - 0.8800015) < 0.00001);
	assert(fabs(weight_neg_1 - -1.1117288) < 0.00001);

	ternarize<num_neurons, num_neurons>(weight_tensor_1, weight_pos_1, weight_neg_1, 0.1f);
	unique_ptr<float[]> weight_tensor_1_t = transpose<num_neurons, num_neurons>(weight_tensor_1);

	unique_ptr<sparse_list_tuple[]> sparse_lists_1 =
	    createSparseList<num_neurons, num_neurons>(weight_tensor_1_t.get(), weight_pos_1, weight_neg_1);

	assert(num_neurons == parameters_npz["fc1/b:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn1/beta:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn1/gamma:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn1/mean/EMA:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn1/variance/EMA:0"].shape[0]);

	float* bias_1     = parameters_npz["fc1/b:0"].data<float>();
	float* beta_1     = parameters_npz["bn1/beta:0"].data<float>();
	float* gamma_1    = parameters_npz["bn1/gamma:0"].data<float>();
	float* mean_1     = parameters_npz["bn1/mean/EMA:0"].data<float>();
	float* variance_1 = parameters_npz["bn1/variance/EMA:0"].data<float>();


	assert(num_neurons == parameters_npz["fc2/W:0"].shape[0]);
	assert(num_neurons == parameters_npz["fc2/W:0"].shape[1]);

	float* weight_tensor_2 = parameters_npz["fc2/W:0"].data<float>();

	float weight_pos_2 = *parameters_npz["fc2/Wp:0"].data<float>();
	float weight_neg_2 = -(*parameters_npz["fc2/Wn:0"].data<float>());

	assert(fabs(weight_pos_2 - 0.86801404) < 0.00001);
	assert(fabs(weight_neg_2 - -1.119899) < 0.00001);

	ternarize<num_neurons, num_neurons>(weight_tensor_2, weight_pos_2, weight_neg_2, 0.1f);
	unique_ptr<float[]> weight_tensor_2_t = transpose<num_neurons, num_neurons>(weight_tensor_2);

	unique_ptr<sparse_list_tuple[]> sparse_lists_2 =
	    createSparseList<num_neurons, num_neurons>(weight_tensor_2_t.get(), weight_pos_2, weight_neg_2);

	assert(num_neurons == parameters_npz["fc2/b:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn2/beta:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn2/gamma:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn2/mean/EMA:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn2/variance/EMA:0"].shape[0]);

	float* bias_2     = parameters_npz["fc2/b:0"].data<float>();
	float* beta_2     = parameters_npz["bn2/beta:0"].data<float>();
	float* gamma_2    = parameters_npz["bn2/gamma:0"].data<float>();
	float* mean_2     = parameters_npz["bn2/mean/EMA:0"].data<float>();
	float* variance_2 = parameters_npz["bn2/variance/EMA:0"].data<float>();


	assert(num_neurons == parameters_npz["fc3/W:0"].shape[0]);
	assert(num_units == parameters_npz["fc3/W:0"].shape[1]);

	float* weight_tensor_3 = parameters_npz["fc3/W:0"].data<float>();

	assert(num_units == parameters_npz["fc3/b:0"].shape[0]);
	float* bias_3 = parameters_npz["fc3/b:0"].data<float>();

	uint n_test_set = dataset.test_images.size();

	for(uint batch = 0; batch < 1 /*n_test_set / batch_size*/; ++batch)
	{
		unique_ptr<float[]> input = init_input<batch_size>(dataset, batch);
		print_frame<28 * batch_size, 28>(input.get());
		normalize<batch_size, frame_size>(input.get());
	}
}

template<uint batch_size>
unique_ptr<float[]> init_input(const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>& data, uint offset)
{
	unique_ptr<float[]> out(new float[batch_size * frame_size]);
	for(uint i = 0; i < batch_size; ++i)
	{
		assert(data.test_images.at(offset * batch_size + i).size() == frame_size);
		copy(data.test_images.at(offset * batch_size + i).begin(), data.test_images.at(i).end(),
		     out.get() + i * frame_size);
	}
	return out;
}

template<uint batch_size, uint frame_size>
void normalize(float* buf)
{
	for(float* ptr = buf; ptr < buf + batch_size * frame_size; ++ptr)
	{
		if(!(*ptr >= 0 && *ptr <= 255))
		{
			std::cout << batch_size * frame_size << '\n';
			std::cout << ptr - buf << '\n';
			std::cout << *ptr << '\n';
		}
		assert(*ptr >= 0 && *ptr <= 255);
		*ptr = *ptr / 255;
	}
}
