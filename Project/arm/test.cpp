#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <cnpy.h>
#include <mnist/mnist_reader.hpp>
#include <mnist/mnist_utils.hpp>

typedef unsigned int uint;
#define log(var) std::cout << #var "=" << var << "\n";

using std::unique_ptr;
using std::vector;

#include "mat.hpp"
#include "sparse.hpp"

const uint batch_size  = 64;
const uint frame_size  = 28 * 28;
const uint num_neurons = 1024;

int main(int argc, char* argv[])
{
	mnist::MNIST_dataset<vector, vector<float>, uint8_t> dataset =
	    mnist::read_dataset<vector, vector, float, uint8_t>("./mnist/");

	cnpy::npz_t parameters_npz = cnpy::npz_load("trained_models/mnist-t40-final.npz");

	assert(parameters_npz["fc0/W:0"].shape[0] == frame_size);
	assert(parameters_npz["fc0/W:0"].shape[1] == num_neurons);

	float* weight_tensor_0 = parameters_npz["fc0/W:0"].data<float>();

	float weight_pos_0 = *parameters_npz["fc0/Wp:0"].data<float>();
	float weight_neg_0 = -(*parameters_npz["fc0/Wn:0"].data<float>());

	assert(fabs(weight_pos_0 - 0.7502421) < 0.00001);
	assert(fabs(weight_neg_0 - -1.2224674) < 0.00001);

	ternarize<frame_size, num_neurons>(weight_tensor_0, weight_pos_0, weight_neg_0, 0.1f);
	unique_ptr<float[]> weight_tensor_0_t = transpose<frame_size, num_neurons>(weight_tensor_0);

	unique_ptr<sparse_list_tuple[]> sparse_lists_0 =
	    createSparseList<num_neurons, frame_size>(weight_tensor_0_t.get(), weight_pos_0, weight_neg_0);
}
