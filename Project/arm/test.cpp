#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>

#include <cnpy.h>
#include <mnist/mnist_reader.hpp>
#include <mnist/mnist_utils.hpp>

#include "mat.hpp"
#include "sparse.hpp"
#include "util.hpp"

#include <arm_neon.h>

using std::copy;

template<uint batch_size>
unique_ptr<float[]> init_input(const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>& data, uint id);

template<uint batch_size>
unique_ptr<int[]> init_labels(const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>& data, uint id);

int main(int argc, char* argv[])
{
	LOG(batch_size);

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
	// unique_ptr<float[]> weight_tensor_0_t = transpose<frame_size, num_neurons>(weight_tensor_0);

	unique_ptr<sparse_list_tuple[]> sparse_lists_0 =
	    createSparseList<frame_size, num_neurons>(weight_tensor_0, weight_pos_0, weight_neg_0);

	assert(num_neurons == parameters_npz["fc0/b:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn0/beta:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn0/gamma:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn0/mean/EMA:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn0/variance/EMA:0"].shape[0]);

	// float* bias_0     = parameters_npz["fc0/b:0"].data<float>();
	float* beta_0     = parameters_npz["bn0/beta:0"].data<float>();
	float* gamma_0    = parameters_npz["bn0/gamma:0"].data<float>();
	float* mean_0     = parameters_npz["bn0/mean/EMA:0"].data<float>();
	float* variance_0 = parameters_npz["bn0/variance/EMA:0"].data<float>();

	unique_ptr<float[]> zeta_0 = compute_zeta<num_neurons>(gamma_0, variance_0);

	assert(num_neurons == parameters_npz["fc1/W:0"].shape[0]);
	assert(num_neurons == parameters_npz["fc1/W:0"].shape[1]);

	float* weight_tensor_1 = parameters_npz["fc1/W:0"].data<float>();

	float weight_pos_1 = *parameters_npz["fc1/Wp:0"].data<float>();
	float weight_neg_1 = -(*parameters_npz["fc1/Wn:0"].data<float>());

	assert(fabs(weight_pos_1 - 0.8800015) < 0.00001);
	assert(fabs(weight_neg_1 - -1.1117288) < 0.00001);

	ternarize<num_neurons, num_neurons>(weight_tensor_1, weight_pos_1, weight_neg_1, 0.1f);
	// unique_ptr<float[]> weight_tensor_1_t = transpose<num_neurons, num_neurons>(weight_tensor_1);

	unique_ptr<sparse_list_tuple[]> sparse_lists_1 =
	    createSparseList<num_neurons, num_neurons>(weight_tensor_1, weight_pos_1, weight_neg_1);

	assert(num_neurons == parameters_npz["fc1/b:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn1/beta:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn1/gamma:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn1/mean/EMA:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn1/variance/EMA:0"].shape[0]);

	// float* bias_1     = parameters_npz["fc1/b:0"].data<float>();
	float* beta_1     = parameters_npz["bn1/beta:0"].data<float>();
	float* gamma_1    = parameters_npz["bn1/gamma:0"].data<float>();
	float* mean_1     = parameters_npz["bn1/mean/EMA:0"].data<float>();
	float* variance_1 = parameters_npz["bn1/variance/EMA:0"].data<float>();

	unique_ptr<float[]> zeta_1 = compute_zeta<num_neurons>(gamma_1, variance_1);

	assert(num_neurons == parameters_npz["fc2/W:0"].shape[0]);
	assert(num_neurons == parameters_npz["fc2/W:0"].shape[1]);

	float* weight_tensor_2 = parameters_npz["fc2/W:0"].data<float>();

	float weight_pos_2 = *parameters_npz["fc2/Wp:0"].data<float>();
	float weight_neg_2 = -(*parameters_npz["fc2/Wn:0"].data<float>());

	assert(fabs(weight_pos_2 - 0.86801404) < 0.00001);
	assert(fabs(weight_neg_2 - -1.119899) < 0.00001);

	ternarize<num_neurons, num_neurons>(weight_tensor_2, weight_pos_2, weight_neg_2, 0.1f);
	// unique_ptr<float[]> weight_tensor_2_t = transpose<num_neurons, num_neurons>(weight_tensor_2);

	unique_ptr<sparse_list_tuple[]> sparse_lists_2 =
	    createSparseList<num_neurons, num_neurons>(weight_tensor_2, weight_pos_2, weight_neg_2);

	assert(num_neurons == parameters_npz["fc2/b:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn2/beta:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn2/gamma:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn2/mean/EMA:0"].shape[0]);
	assert(num_neurons == parameters_npz["bn2/variance/EMA:0"].shape[0]);

	// float* bias_2     = parameters_npz["fc2/b:0"].data<float>();
	float* beta_2     = parameters_npz["bn2/beta:0"].data<float>();
	float* gamma_2    = parameters_npz["bn2/gamma:0"].data<float>();
	float* mean_2     = parameters_npz["bn2/mean/EMA:0"].data<float>();
	float* variance_2 = parameters_npz["bn2/variance/EMA:0"].data<float>();

	unique_ptr<float[]> zeta_2 = compute_zeta<num_neurons>(gamma_2, variance_2);

	assert(num_neurons == parameters_npz["fc3/W:0"].shape[0]);
	assert(num_units == parameters_npz["fc3/W:0"].shape[1]);

	float* weight_tensor_3 = parameters_npz["fc3/W:0"].data<float>();

	assert(num_units == parameters_npz["fc3/b:0"].shape[0]);
	// float* bias_3 = parameters_npz["fc3/b:0"].data<float>();

	uint n_test_set = dataset.test_images.size();

	uint num_batches = n_test_set / batch_size;
	float accuracy   = 0;

	unique_ptr<unique_ptr<float[]>[]> inputs(new unique_ptr<float[]>[num_batches]);
	unique_ptr<unique_ptr<int[]>[]> labels(new unique_ptr<int[]>[num_batches]);

	for(uint batch = 0; batch < num_batches; ++batch)
	{
		// unique_ptr<float[]> input = init_input<batch_size>(dataset, batch);
		// normalize<batch_size, frame_size>(input.get());
		//
		// inputs_t[batch] = transpose<batch_size, frame_size>(input.get());
		inputs[batch] = init_input<batch_size>(dataset, batch);
		normalize<batch_size, frame_size>(inputs[batch].get());
		labels[batch] = init_labels<batch_size>(dataset, batch);
	}

	auto t0 = std::chrono::high_resolution_clock::now();
	for(uint batch = 0; batch < num_batches; ++batch)
	{
		unique_ptr<float[]>& input = inputs[batch];
		unique_ptr<int[]>& label   = labels[batch];

		// Fist Layer
		unique_ptr<float[]> out_tensor_0 = sparseMatrixMultiply<batch_size, frame_size, num_neurons>(
		    input.get(), sparse_lists_0.get(), weight_pos_0, weight_neg_0);
		// unique_ptr<float[]> out_tensor_0 = mul<batch_size, frame_size, num_neurons>(input.get(), weight_tensor_0);

		batch_normalization_arm<batch_size, num_neurons>(out_tensor_0.get(), mean_0, beta_0, zeta_0.get());
		ReLU<batch_size, num_neurons>(out_tensor_0.get(), 0.0);

		// Second Layer
		unique_ptr<float[]> out_tensor_1 = sparseMatrixMultiply<batch_size, num_neurons, num_neurons>(
		    out_tensor_0.get(), sparse_lists_1.get(), weight_pos_1, weight_neg_1);
		// unique_ptr<float[]> out_tensor_1 =
		//     mul<batch_size, num_neurons, num_neurons>(out_tensor_0.get(), weight_tensor_1);

		batch_normalization_arm<batch_size, num_neurons>(out_tensor_1.get(), mean_1, beta_1, zeta_1.get());
		ReLU<batch_size, num_neurons>(out_tensor_1.get(), 0.0);

		// Third Layer
		unique_ptr<float[]> out_tensor_2 = sparseMatrixMultiply<batch_size, num_neurons, num_neurons>(
		    out_tensor_1.get(), sparse_lists_2.get(), weight_pos_2, weight_neg_2);
		// unique_ptr<float[]> out_tensor_2 =
		//     mul<batch_size, num_neurons, num_neurons>(out_tensor_1.get(), weight_tensor_2);

		batch_normalization_arm<batch_size, num_neurons>(out_tensor_2.get(), mean_2, beta_2, zeta_2.get());
		ReLU<batch_size, num_neurons>(out_tensor_2.get(), 0.0);

		// unique_ptr<float[]> out_tensor_2 = transpose<num_neurons, batch_size>(out_tensor_2_t.get());

		unique_ptr<float[]> out_tensor_3 = mul<batch_size, num_neurons, num_units>(out_tensor_2.get(), weight_tensor_3);

		Softmax<batch_size, num_units>(out_tensor_3.get());
		// for(uint i = 0; i < batch_size; ++i)
		// {
		// 	print_list<num_units>(out_tensor_3.get() + i * num_units);
		// 	printf("==> %d\n", max_id<num_units>(out_tensor_3.get() + i * num_units));
		// }

		float accuracy_batch = get_accuracy<batch_size, num_units>(out_tensor_3.get(), label.get());
		accuracy += accuracy_batch;
	}
	auto t1     = std::chrono::high_resolution_clock::now();
	double time = std::chrono::duration<double>(t1 - t0).count();
	LOG(time);

	accuracy /= num_batches;
	LOG(accuracy);
}

template<uint batch_size>
unique_ptr<float[]> init_input(const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>& data, uint offset)
{
	unique_ptr<float[]> out(new float[batch_size * frame_size]);
	for(uint i = 0; i < batch_size; ++i)
	{
		uint idx = offset * batch_size + i;
		assert(data.test_images.at(idx).size() == frame_size);
		copy(data.test_images.at(idx).begin(), data.test_images.at(idx).end(), out.get() + i * frame_size);
	}
	return out;
}

template<uint batch_size>
unique_ptr<int[]> init_labels(const mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>& data, uint id)
{
	unique_ptr<int[]> out(new int[batch_size * frame_size]);
	for(int frame_id = 0; frame_id < batch_size; frame_id++)
		out[frame_id] = data.test_labels.at(id * batch_size + frame_id);
	return out;
}
