#include <iostream>
#include <time.h>       /* time */
#include "../../third_party/mnist/include/mnist/mnist_reader.hpp"
#include "../../third_party/mnist/include/mnist/mnist_utils.hpp"
#include "../../third_party/cnpy/cnpy.h"

#include <math.h>       /* sqrt */

 /*
  *
  * Compile: g++ -o mnist main.cpp -L../../build/ -lcnpy --std=c++11 -Wall -O3
  *
  */

#define MNIST_DATA_LOCATION "../../third_party/mnist/"

// MLP configuration
#define BATCH_SIZE 64
#define FRAME_SIZE 28*28
#define NUM_NEURONS 1024
#define NUM_UNITS 10

//#define PRINT_STATS
//#define EIGEN_DONT_VECTORIZE
#include "../../third_party/Eigen/Eigen/Dense"
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrixXf;

void FullyConnected( float* InputTensor, float* WeightTensor, float* Output, float* bias, int m, int n, int k ) {
  RMatrixXf A = Eigen::Map<RMatrixXf>( InputTensor, m, n  );
  RMatrixXf B = Eigen::Map<RMatrixXf>( WeightTensor, n, k );
  RMatrixXf C = RMatrixXf::Zero( m, k );

  C.noalias() += A * B;

  Eigen::Map<RMatrixXf>( Output, m, k ) = C;

  // Very simple matrix multiply
  /*for( auto i=0; i<m; ++i ) {
    for( auto j=0; j<k; ++j ) {
      float result=0.0;
      for( auto kk=0; kk<n; ++kk ) {
        result += InputTensor[i*n+kk] * WeightTensor[kk*k+j];
      }
      Output[i*k+j] = result + bias[j];
    }
  }*/
}

int ReLU( float* InputTensor, int x, int y, float threshold ) {
  double zero_count=0;
  for( int i=0; i<x*y; ++i ) {
    InputTensor[i] = (InputTensor[i] < threshold) ? 0 : InputTensor[i];
    if( InputTensor[i]==0 ) zero_count++;
  }
/*#ifdef PRINT_STATS
  double sparsity = (double)(zero_count/(x*y))*100;
  std::cout << "[STATS] Found " << sparsity << "% sparsity in batch." << std::endl;
#endif*/
  return (int) zero_count;
}

void checkSparsity( float* InputTensor, int x, int y ) {
  double zero_count;
  for( int j=0; j<y; ++j ) {
    zero_count=0.0;
    for( int i=0; i<x; ++i ) {
      if( InputTensor[i*y+j]==0 ) zero_count++;
    }
    double sparsity = (double)(zero_count/(x))*100;
    //std::cout << "[STATS] Found " << sparsity << "% sparsity in col." << std::endl;
  }
}

void Batchnormalization( float* InputTensor, int x, int y, float* beta, float* gamma, float* mean, float* variance ) {
  assert( y==NUM_NEURONS );
  for( int i=0; i<x; ++i ) {
    for( int j=0; j<y; ++j ) {
      // TODO: gamma/sqrt(variance+epsilon) can be pre-computed
      InputTensor[i*y+j] = ((InputTensor[i*y+j] - mean[j])*gamma[j])/sqrt(variance[j]+1e-4) + beta[j];
    }
  }
}
/*
void Batchnormalization( float* InputTensor, int x, int y, float* beta, float* gamma, float* mean, float* variance ) {
  assert( y==NUM_NEURONS );
  for( int i=0; i<x; ++i ) {
    for( int j=0; j<y; ++j ) {
      // TODO: gamma/sqrt(variance+epsilon) can be pre-computed
      InputTensor[i*y+j] = ((InputTensor[i*y+j] - mean[j])*gamma[j])/sqrt(variance[j]+1e-4) + beta[j];
    }
  }
}*/

void Softmax( float* logits, int batch_size, int num_units ) {
  float max, sum;

  for( int id=0; id<batch_size; ++id ) {
    max = 0.0;
    sum = 0.0;
    for( int i = 0; i < num_units; i++ ) if (max < logits[id*num_units+i]) max = logits[id*num_units+i];
    for( int i=0; i<num_units; ++i ) {
      logits[id*num_units+i] = exp(logits[id*num_units+i] - max);
      sum += logits[id*num_units+i];
    }
    for( int i=0; i<num_units; ++i ) {
      logits[id*num_units+i] /= sum;
    }
  }
}

float get_accuracy( float* probs, int* labels, int batch_size, int num_units ) {
  float max;
  int pred_class;
  int correct=0;

  for( int id=0; id<batch_size; ++id ) {
    // Get class with highest probability
    max=0.0;
    pred_class=0;
    for( int i = 0; i < num_units; i++ ) {
      if (max < probs[id*num_units+i]) {
        max = probs[id*num_units+i];
        pred_class = i;
      }
    }
    // Check against label
    if( pred_class==labels[id] ) correct++;
  }
#ifdef PRINT_STATS
  std::cout << "[STATS] Correct = " << correct << "/" << batch_size << std::endl;
#endif

  return (float) correct/batch_size;
}

int initialize_input( float* InputTensor, mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>& dataset, int batch_size, int batch_id ) {
  double zero_count=0;

  int frame_size = dataset.test_images.at(0).size();
  for( int frame_id=0; frame_id<batch_size; frame_id++ ) {
    for( int pixel_id=0; pixel_id<frame_size; ++pixel_id ) {
      InputTensor[frame_id*frame_size + pixel_id] = (float) dataset.test_images.at(batch_id*batch_size+frame_id).at(pixel_id);
      if( InputTensor[frame_id*frame_size + pixel_id]==0 ) zero_count++;
    }
  }
#ifdef PRINT_STATS
  double sparsity = (double)(zero_count/(batch_size*frame_size))*100;
  std::cout << "[STATS] Found " << sparsity << "% sparsity in input batch, " << zero_count << "/" << batch_size*frame_size << std::endl;
#endif
  return (int) zero_count;
}

void initialize_labels( int* labels, mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t>& dataset, int batch_size, int batch_id ) {
  for( int frame_id=0; frame_id<batch_size; frame_id++ ) {
    labels[frame_id] = (int) dataset.test_labels.at(batch_id*batch_size+frame_id);
  }
}

void normalize_pixel( float* InputTensor, int batch_size, int frame_size ) {
  for( int pixel_id=0; pixel_id<batch_size*frame_size; ++pixel_id ) {
    if( InputTensor[pixel_id]<0 || InputTensor[pixel_id]>255 ) std::cout << "Pixel out of range!" << std::endl;
    InputTensor[pixel_id] = (float) (InputTensor[pixel_id]/255.0);
  }
}

void center_pixel_to_zero( float* InputTensor, int batch_size, int frame_size ) {
  for( int pixel_id=0; pixel_id<batch_size*frame_size; ++pixel_id ) {
    InputTensor[pixel_id] = (float) InputTensor[pixel_id]*2.0-1.0;
  }
}

int main(int argc, char* argv[]) {
  std::cout << "[INFO] MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_seconds;
  double overall_time=0.0;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<float>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, float, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "[INFO] Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "[INFO] Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "[INFO] Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "[INFO] Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    /* Initialize all parameters from a numpy file */
    cnpy::npz_t parameters_npz = cnpy::npz_load("../../trained_models/mnist-float32.npz");
    //cnpy::npz_t parameters_npz = cnpy::npz_load("./mnist-float32-nn.npz");

    assert( FRAME_SIZE == parameters_npz["fc0/W:0"].shape[0] );
    assert( NUM_NEURONS == parameters_npz["fc0/W:0"].shape[1] );
    float *WeightTensor0 = parameters_npz["fc0/W:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["fc0/b:0"].shape[0] );
    float *bias0 = parameters_npz["fc0/b:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn0/beta:0"].shape[0] );
    float *beta0 = parameters_npz["bn0/beta:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn0/gamma:0"].shape[0] );
    float *gamma0 = parameters_npz["bn0/gamma:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn0/mean/EMA:0"].shape[0] );
    float *mean0 = parameters_npz["bn0/mean/EMA:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn0/variance/EMA:0"].shape[0] );
    float *variance0 = parameters_npz["bn0/variance/EMA:0"].data<float>();

    assert( NUM_NEURONS == parameters_npz["fc1/W:0"].shape[0] );
    assert( NUM_NEURONS == parameters_npz["fc1/W:0"].shape[1] );
    float *WeightTensor1 = parameters_npz["fc1/W:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["fc1/b:0"].shape[0] );
    float *bias1 = parameters_npz["fc1/b:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn1/beta:0"].shape[0] );
    float *beta1 = parameters_npz["bn1/beta:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn1/gamma:0"].shape[0] );
    float *gamma1 = parameters_npz["bn1/gamma:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn1/mean/EMA:0"].shape[0] );
    float *mean1 = parameters_npz["bn1/mean/EMA:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn1/variance/EMA:0"].shape[0] );
    float *variance1 = parameters_npz["bn1/variance/EMA:0"].data<float>();

    assert( NUM_NEURONS == parameters_npz["fc2/W:0"].shape[0] );
    assert( NUM_NEURONS == parameters_npz["fc2/W:0"].shape[1] );
    float *WeightTensor2 = parameters_npz["fc2/W:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["fc2/b:0"].shape[0] );
    float *bias2 = parameters_npz["fc2/b:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn2/beta:0"].shape[0] );
    float *beta2 = parameters_npz["bn2/beta:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn2/gamma:0"].shape[0] );
    float *gamma2 = parameters_npz["bn2/gamma:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn2/mean/EMA:0"].shape[0] );
    float *mean2 = parameters_npz["bn2/mean/EMA:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn2/variance/EMA:0"].shape[0] );
    float *variance2 = parameters_npz["bn2/variance/EMA:0"].data<float>();

    assert( NUM_NEURONS == parameters_npz["fc3/W:0"].shape[0] );
    assert( NUM_UNITS == parameters_npz["fc3/W:0"].shape[1] );
    float *WeightTensor3 = parameters_npz["fc3/W:0"].data<float>();
    assert( NUM_UNITS == parameters_npz["fc3/b:0"].shape[0] );
    float *bias3 = parameters_npz["fc3/b:0"].data<float>();

    /* Allocate output tensors */
    float *OutputTensor0 = (float*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(float) );
    assert( OutputTensor0 != NULL );
    float *OutputTensor1 = (float*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(float) );
    assert( OutputTensor1 != NULL );
    float *OutputTensor2 = (float*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(float) );
    assert( OutputTensor2 != NULL );
    float *logits = (float*)malloc( BATCH_SIZE * NUM_UNITS * sizeof(float) );
    assert( logits != NULL );

    int size_test_set = dataset.test_images.size();
    float *Input0 = (float*)malloc( BATCH_SIZE * FRAME_SIZE * sizeof(float) );
    assert( Input0 != NULL );
    int *labels = (int*)malloc( BATCH_SIZE * sizeof(int) );
    assert( labels != NULL );

    float accuracy=0.0;
    int num_batches=0;
    unsigned long long int zero_count=0;
    unsigned long long int activation_count=0;
    float threshold = std::atof(argv[1]);
    /*
     *
     * This is the interesting part
     *
     */
    for( int batch_id=0; batch_id<size_test_set/BATCH_SIZE; ++batch_id ) {
      /* Initialize InputTensor with a batch of frames */
      zero_count += (unsigned long long int) initialize_input( Input0, dataset, BATCH_SIZE, batch_id );
      activation_count += (unsigned long long int) BATCH_SIZE * FRAME_SIZE;
      initialize_labels( labels, dataset, BATCH_SIZE, batch_id );
      normalize_pixel( Input0, BATCH_SIZE, FRAME_SIZE );
      center_pixel_to_zero( Input0, BATCH_SIZE, FRAME_SIZE );

      start = std::chrono::system_clock::now();
      // Layer 0
      FullyConnected( Input0, WeightTensor0, OutputTensor0, bias0, BATCH_SIZE, FRAME_SIZE, NUM_NEURONS );
      Batchnormalization( OutputTensor0, BATCH_SIZE, NUM_NEURONS, beta0, gamma0, mean0, variance0 );
      zero_count += (unsigned long long int) ReLU( OutputTensor0, BATCH_SIZE, NUM_NEURONS, 0.00 );
      //checkSparsity( OutputTensor0, BATCH_SIZE, NUM_NEURONS );
      activation_count += (unsigned long long int) BATCH_SIZE * NUM_NEURONS;

      // Layer 1
      FullyConnected( OutputTensor0, WeightTensor1, OutputTensor1, bias1, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS );
      Batchnormalization( OutputTensor1, BATCH_SIZE, NUM_NEURONS, beta1, gamma1, mean1, variance1 );
      zero_count += (unsigned long long int) ReLU( OutputTensor1, BATCH_SIZE, NUM_NEURONS, 0.00);
      //checkSparsity( OutputTensor1, BATCH_SIZE, NUM_NEURONS );
      activation_count += (unsigned long long int) BATCH_SIZE * NUM_NEURONS;

      // Layer 2
      FullyConnected( OutputTensor1, WeightTensor2, OutputTensor2, bias2, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS );
      Batchnormalization( OutputTensor2, BATCH_SIZE, NUM_NEURONS, beta2, gamma2, mean2, variance2 );
      zero_count += (unsigned long long int) ReLU( OutputTensor2, BATCH_SIZE, NUM_NEURONS, 0.0 );
      activation_count += (unsigned long long int) BATCH_SIZE * NUM_NEURONS;

      // Layer 3
      FullyConnected( OutputTensor2, WeightTensor3, logits, bias3, BATCH_SIZE, NUM_NEURONS, NUM_UNITS );
      Softmax( logits, BATCH_SIZE, NUM_UNITS );

      end = std::chrono::system_clock::now();
      elapsed_seconds = end-start;
      overall_time += elapsed_seconds.count();

      float accuracy_batch = get_accuracy( logits, labels, BATCH_SIZE, NUM_UNITS )*100;
      accuracy += accuracy_batch;
      num_batches++;
    }

    accuracy = accuracy/num_batches;
    float sparsity = ((float)zero_count/(float)activation_count)*100;
    std::cout << "[RESULT] Sparsity       = " << sparsity << "%, " << zero_count << "/" << activation_count << std::endl;
    std::cout << "[RESULT] Accuracy       = " << accuracy << "%" << std::endl;
    std::cout << "[RESULT] Inference Rate = " << size_test_set/overall_time << " FPS" << std::endl;


    // Release all allocated memory
    free(Input0);
    free(OutputTensor0);
    free(OutputTensor1);
    free(OutputTensor2);
    free(logits);

    return 0;
}
