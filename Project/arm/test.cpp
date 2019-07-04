#include <algorithm>
#include <cassert>
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

template<uint m, uint n>
void print_frame(float* frame)
{
	for(uint i = 0; i < m; ++i)
	{
		for(uint j = 0; j < n; ++j) printf(frame[i * n + j] == 0 ? " " : "#");
		printf("\n");
	}
}

template<uint n, typename T>
void print_list(T* list)
{
	std::cout << "[ ";
	for(uint i = 0; i < n; ++i) std::cout << list[i] << ", ";
	std::cout << "]\n";
}

template<uint n, typename T>
uint max_id(T* list)
{
	uint id = 0;
	T max   = *list;
	for(uint i = 1; i < n; ++i)
	{
		T value = list[i];
		if(value > max)
		{
			id  = i;
			max = value;
		};
	}
	return id;
}

template<uint m, uint n>
void print_mat(float* a)
{	
	for(uint i = 0; i < m; ++i)
	{
		for(uint j = 0; j < n; ++j) printf("%.2f \t", a[i * n + j]);
		printf("\n");
	}
	printf("\n");
}

void print_sparse_list(sparse_list& list)
{
	for(uint i = 0; i < list.n; ++i)
	{
		printf("%d, ", list.i[i]);
	} 
}

template<uint n>
void print_sparse_list(sparse_list_tuple* tuple)
{
	for(uint i = 0; i < n; ++i)
	{
		printf("pos: ");
		print_sparse_list(tuple->pos);
		printf("\nneg: ");
		print_sparse_list(tuple->neg);
		printf("\n");
		++tuple;
	}
	printf("\n");
}

template<uint row>
void print_sparse_listJona(sparse_list_tuple* tuple)
{
	for(uint i = 0; i < row; ++i)
	{
		printf("Col %d: \n", i);
		printf("\tpos: ");
		print_sparse_list(tuple->pos);
		printf("\n\tneg: ");
		print_sparse_list(tuple->neg);
		printf("\n");
		++tuple;
	}
	printf("\n");
}

template<uint row, uint col>
void print_matJona(float* a)
{	
	for(uint i = 0; i < row; ++i)
	{
		for(uint j = 0; j < col; ++j){
			printf("%.2f \t", a[i * col + j]);
		} 
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char* argv[])
{
#ifdef TEST

	// Create Input Matrix
	float input[] { 5, 7, 3, 1, 3, 1, 5, 9 };
	printf("Input Mat:\n");
	print_matJona<batch_size, frame_size>(input);


	//Create weight 0
	float weight_tensor_0_t[] { 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1 };
	printf("Weight 0 Mat:\n");
	print_matJona<frame_size, num_neurons>(weight_tensor_0_t);

	//Create Sparse List weight 0
	unique_ptr<sparse_list_tuple[]> sparse_lists_0 = createSparseListJona<frame_size, num_neurons>(weight_tensor_0_t, 1, -1);
	printf("Weight 0 Sparse List:\n");
	print_sparse_listJona<num_neurons>(sparse_lists_0.get());

	//Create weight 1
	float weight_tensor_1_t[] { 0, 1, 0, 1, 0, 1 };
	printf("Weight 1 Mat:\n");
	print_matJona<num_neurons, num_units>(weight_tensor_1_t);

	//Create Sparse List weight 1
	unique_ptr<sparse_list_tuple[]> sparse_lists_1 = createSparseListJona<num_neurons, num_units>(weight_tensor_1_t, 1, -1);
	printf("Weight 1 Sparse List:\n");
	print_sparse_listJona<num_units>(sparse_lists_1.get());


	//First sparse Matrix Multiply
	unique_ptr<float[]> _out_tensor_0 = mulJona<batch_size, frame_size, num_neurons>(input, weight_tensor_0_t);
	printf("Output 0 normal:\n");
	print_matJona<batch_size, num_neurons>(_out_tensor_0.get());

	unique_ptr<float[]> out_tensor_0 = sparseMatrixMultiplyJona<batch_size, frame_size, num_neurons>(input, sparse_lists_0.get(), 1, -1);
	printf("Output 0 Sparse:\n");
	print_matJona<batch_size, num_neurons>(out_tensor_0.get());


	//Second sparse Matrix Multiply
	unique_ptr<float[]> _out_tensor_1 = mulJona<batch_size, num_neurons, num_units>(out_tensor_0.get(), weight_tensor_1_t);
	printf("Output 1 normal:\n");
	print_mat<batch_size, num_units>(_out_tensor_1.get());

	unique_ptr<float[]> out_tensor_1 = sparseMatrixMultiplyJona<batch_size, num_neurons, num_units>(out_tensor_0.get(), sparse_lists_1.get(), 1, -1);
	printf("Output 1 Sparse:\n");
	print_mat<batch_size, num_units>(out_tensor_1.get());

#else
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
	//unique_ptr<float[]> weight_tensor_0_t = transpose<frame_size, num_neurons>(weight_tensor_0);

	unique_ptr<sparse_list_tuple[]> sparse_lists_0 = createSparseListJona<frame_size, num_neurons>(weight_tensor_0, weight_pos_0, weight_neg_0);

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

	unique_ptr<float[]> zeta_0 = compute_zeta<num_neurons>(gamma_0, variance_0);

	assert(num_neurons == parameters_npz["fc1/W:0"].shape[0]);
	assert(num_neurons == parameters_npz["fc1/W:0"].shape[1]);

	float* weight_tensor_1 = parameters_npz["fc1/W:0"].data<float>();

	float weight_pos_1 = *parameters_npz["fc1/Wp:0"].data<float>();
	float weight_neg_1 = -(*parameters_npz["fc1/Wn:0"].data<float>());

	assert(fabs(weight_pos_1 - 0.8800015) < 0.00001);
	assert(fabs(weight_neg_1 - -1.1117288) < 0.00001);

	ternarize<num_neurons, num_neurons>(weight_tensor_1, weight_pos_1, weight_neg_1, 0.1f);
	//unique_ptr<float[]> weight_tensor_1_t = transpose<num_neurons, num_neurons>(weight_tensor_1);

	unique_ptr<sparse_list_tuple[]> sparse_lists_1 = createSparseListJona<num_neurons, num_neurons>(weight_tensor_1, weight_pos_1, weight_neg_1);

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

	unique_ptr<float[]> zeta_1 = compute_zeta<num_neurons>(gamma_1, variance_1);

	assert(num_neurons == parameters_npz["fc2/W:0"].shape[0]);
	assert(num_neurons == parameters_npz["fc2/W:0"].shape[1]);

	float* weight_tensor_2 = parameters_npz["fc2/W:0"].data<float>();

	float weight_pos_2 = *parameters_npz["fc2/Wp:0"].data<float>();
	float weight_neg_2 = -(*parameters_npz["fc2/Wn:0"].data<float>());

	assert(fabs(weight_pos_2 - 0.86801404) < 0.00001);
	assert(fabs(weight_neg_2 - -1.119899) < 0.00001);

	ternarize<num_neurons, num_neurons>(weight_tensor_2, weight_pos_2, weight_neg_2, 0.1f);
	//unique_ptr<float[]> weight_tensor_2_t = transpose<num_neurons, num_neurons>(weight_tensor_2);

	unique_ptr<sparse_list_tuple[]> sparse_lists_2 = createSparseListJona<num_neurons, num_neurons>(weight_tensor_2, weight_pos_2, weight_neg_2);

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

	unique_ptr<float[]> zeta_2 = compute_zeta<num_neurons>(gamma_2, variance_2);

	assert(num_neurons == parameters_npz["fc3/W:0"].shape[0]);
	assert(num_units == parameters_npz["fc3/W:0"].shape[1]);

	float* weight_tensor_3 = parameters_npz["fc3/W:0"].data<float>();

	unique_ptr<float[]> weight_tensor_3_t = transpose<frame_size, num_neurons>(weight_tensor_3);

	assert(num_units == parameters_npz["fc3/b:0"].shape[0]);
	float* bias_3 = parameters_npz["fc3/b:0"].data<float>();

	uint n_test_set = dataset.test_images.size();

	uint num_batches = n_test_set / batch_size;
	uint zero_count  = 0;
	float accuracy   = 0;

	for(uint batch = 0; batch < num_batches; ++batch)
	{
		unique_ptr<float[]> input = init_input<batch_size>(dataset, batch);
		unique_ptr<int[]> labels  = init_labels<batch_size>(dataset, batch);

		//print_frame<28 * batch_size, 28>(input.get());
		//print_list<batch_size>(labels.get());

		normalize<batch_size, frame_size>(input.get());
		//unique_ptr<float[]> input_t = transpose<frame_size, batch_size>(input.get());

		// First Layer
		unique_ptr<float[]> out_tensor_0_t = sparseMatrixMultiplyJona<batch_size, frame_size, num_neurons>(input.get(), sparse_lists_0.get(), weight_pos_0, weight_neg_0);	
		batch_normalizationJona<batch_size, num_neurons>(out_tensor_0_t.get(), beta_0, gamma_0, mean_0, variance_0);
		zero_count += ReLU<num_neurons, batch_size>(out_tensor_0_t.get(), 0.0);	

		// Second Layer
		unique_ptr<float[]> out_tensor_1_t = sparseMatrixMultiplyJona<batch_size, num_neurons, num_neurons>(out_tensor_0_t.get(), sparse_lists_1.get(), weight_pos_1, weight_neg_1);
		batch_normalizationJona<batch_size, num_neurons>(out_tensor_1_t.get(), beta_1, gamma_1, mean_1, variance_1);
		zero_count += ReLU<num_neurons, batch_size>(out_tensor_1_t.get(), 0.0);

		// Third Layer
		unique_ptr<float[]> out_tensor_2_t = sparseMatrixMultiplyJona<batch_size, num_neurons, num_neurons>(out_tensor_1_t.get(), sparse_lists_2.get(), weight_pos_2, weight_neg_2);
		batch_normalizationJona<batch_size, num_neurons>(out_tensor_2_t.get(), beta_2, gamma_2, mean_2, variance_2);
		zero_count += ReLU<num_neurons, batch_size>(out_tensor_2_t.get(), 0.0);

		//unique_ptr<float[]> out_tensor_2 = transpose<num_neurons, batch_size>(out_tensor_2_t.get());

		unique_ptr<float[]> out_tensor_3 = mulJona<batch_size, num_neurons, num_units>(out_tensor_2_t.get(), weight_tensor_3);

		// print_mat<batch_size, num_units>(out_tensor_3.get());

		Softmax<batch_size, num_units>(out_tensor_3.get());
		// for(uint i = 0; i < batch_size; ++i)
		// {
		// 	print_list<num_units>(out_tensor_3.get() + i * num_units);
		// 	printf("==> %d\n", max_id<num_units>(out_tensor_3.get() + i * num_units));
		// }

		int accuracy_batch = get_accuracy<batch_size, num_units>(out_tensor_3.get(), labels.get()) * 100;
		accuracy += accuracy_batch;
		printf(".");
		fflush(stdout);
	}
	accuracy = accuracy / num_batches;
	printf("\n");
	LOG(accuracy);
#endif
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
