#include <iostream>
#include <time.h>       /* time */
#include "../../third_party/mnist/include/mnist/mnist_reader.hpp"
#include "../../third_party/mnist/include/mnist/mnist_utils.hpp"
#include "../../third_party/cnpy/cnpy.h"
// Sparse OPs
#include "../../inc/sparse_ops.h"

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

int oldReLU( float* InputTensor, int x, int y, float threshold ) {
  double zero_count=0;
  for( int i=0; i<x*y; ++i ) {
    InputTensor[i] = (InputTensor[i] < threshold) ? 0 : InputTensor[i];
    //InputTensor[i] = 2*InputTensor[i]-1;
    if( InputTensor[i]==0 ) zero_count++;
  }
#ifdef PRINT_STATS
  double sparsity = (double)(zero_count/(x*y))*100;
  std::cout << "[STATS] Found " << sparsity << "% sparsity in batch." << std::endl;
#endif
  return (int) zero_count;
}

int ReLU( float* InputTensor, int x, int y, float threshold, int bit_a ) {
  float n = 255;// float(pow(2,bit_a) - 1);
  //std::cout << "n=" << n << std::endl;
  double zero_count=0;
  int distance=0;
  for( int i=0; i<x*y; ++i ) {
    if( InputTensor[i] < threshold) {
      InputTensor[i] = 0.0;
    }
    else if( InputTensor[i] > 1) {
      InputTensor[i] = 1.0;
    }
    else InputTensor[i] = (round(InputTensor[i]*n)/n);
    //InputTensor[i] = 2*InputTensor[i]-1;
    //std::cout << "Activation=" << InputTensor[i] << std::endl;
    if( InputTensor[i]==0 ) zero_count++;
  }
#ifdef PRINT_STATS
  double sparsity = (double)(zero_count/(x*y))*100;
  std::cout << "[STATS] Found " << sparsity << "% sparsity in batch." << std::endl;
#endif
  return (int) zero_count;
}

int ReLU( float* InputTensor, uint16_t* QuantTensor, int x, int y, float threshold, int bit_a ) {
  float n = float(pow(2,bit_a)-1);
  //std::cout << "n=" << n << std::endl;
  double zero_count=0;
  //int distance=0;
  for( int i=0; i<x*y; ++i ) {
    if( InputTensor[i] < threshold)  InputTensor[i] = 0.0;
    else if( InputTensor[i] > 1) InputTensor[i] = 1.0;
    QuantTensor[i] = round(InputTensor[i]*n);
    if(QuantTensor[i]>255) std::cout << "[ERROR] Activation out of range." << std::endl;
    //InputTensor[i] = 2*InputTensor[i]-1;
    //std::cout << "Activation=" << InputTensor[i] << std::endl;
    if( InputTensor[i]==0 ) {
      zero_count++;
      //distance++;
    }
    else {
      //std::cout << "distance=" << distance << std::endl;
      //distance=0;
    }
  }
  return (int) zero_count;
}

int ReLU( float* InputTensor, uint16_t* QuantTensor, uint16_t* indices, int x, int y, float threshold, int bit_a ) {
  float n = float(pow(2,bit_a)-1);
  //std::cout << "n=" << n << std::endl;
  double zero_count=0;
  int distance=0;
  uint16_t col_count;
  for( int i=0; i<y; ++i ) {
    col_count=0;
    for( int j=0; j<x; ++j ) {
      if( InputTensor[i*x+j] < threshold) {
        InputTensor[i*x+j] = 0.0;
        //distance++;
      }
      else if( InputTensor[i*x+j] > 1) {
        InputTensor[i*x+j] = 1.0;
      }

      //if(QuantTensor[i*y+j]>255) std::cout << "[ERROR] Activation out of range." << std::endl;
      //InputTensor[i] = 2*InputTensor[i]-1;
      //std::cout << "Activation=" << InputTensor[i] << std::endl;
      if( InputTensor[i*x+j]==0 ) {
        zero_count++;
        distance++;
      }
      else
      {
        std::cout << "distance=" << distance << std::endl;
        distance=0;
        //QuantTensor[i*x+col_count] = round(InputTensor[i*x+j]*n);
        InputTensor[i*x+col_count] = InputTensor[i*x+j];
        // Start with the amount (+1)
        indices[i*(x+1)+col_count+1] = j;
        col_count++;
      }
    }
    indices[i*(x+1)]=col_count;
  }
  return (int) zero_count;
}

int ReLU( float* InputTensor, input_t* QuantTensor, uint16_t* indices, int x, int y, float threshold, int bit_a ) {
  float n = float(pow(2,bit_a)-1);
  //std::cout << "n=" << n << std::endl;
  double zero_count=0;
  uint16_t col_count;
  for( int i=0; i<y; ++i ) {
    col_count=0;
    for( int j=0; j<x; ++j ) {
      if( InputTensor[i*x+j] < threshold) InputTensor[i*x+j] = 0.0;
      else if( InputTensor[i*x+j] > 1) InputTensor[i*x+j] = 1.0;

      //if(QuantTensor[i*y+j]>255) std::cout << "[ERROR] Activation out of range." << std::endl;
      //InputTensor[i] = 2*InputTensor[i]-1;
      //std::cout << "Activation=" << InputTensor[i] << std::endl;
      if( InputTensor[i*x+j]==0 ) {
        zero_count++;
      }
      //else
      {
        //QuantTensor[i*x+col_count] = round(InputTensor[i*x+j]*n);
        QuantTensor[i*(x+2)+col_count+1].val = InputTensor[i*x+j];
        // Start with the amount (+1)
        QuantTensor[i*(x+2)+col_count+1].index = j;
        col_count++;
      }
    }
    QuantTensor[i*(x+1)+0].index = col_count;
    QuantTensor[i*(x+1)+0].val = col_count;
    //indices[i*(x+1)]=col_count;
  }
  return (int) zero_count;
}

void BatchnormalizationRMO( float* InputTensor, int x, int y, float* beta, float* gamma, float* mean, float* variance ) {
  assert( y==NUM_NEURONS );
  for( int i=0; i<x; ++i ) {
    for( int j=0; j<y; ++j ) {
      // TODO: gamma/sqrt(variance+epsilon) can be pre-computed
      InputTensor[i*y+j] = ((InputTensor[i*y+j] - mean[j])*gamma[j])/sqrt(variance[j]+1e-4) + beta[j];
    }
  }
}

void BatchnormalizationCMO( float* InputTensor, int x, int y, float* beta, float* gamma, float* mean, float* variance ) {
  assert( y==NUM_NEURONS );
  for( int i=0; i<y; ++i ) {
    for( int j=0; j<x; ++j ) {
      // TODO: gamma/sqrt(variance+epsilon) can be pre-computed
      InputTensor[i*x+j] = ((InputTensor[i*x+j] - mean[i])*gamma[i])/sqrt(variance[i]+1e-4) + beta[i];
    }
  }
}

/*
 * Precompute zeta = gamma/sqrt(variance+epsilon)
 */
void compute_zeta( float* zeta, int x, float* gamma, float* variance ) {
  for( int i=0; i<x; ++i ) {
    zeta[i] = gamma[i]/sqrt(variance[i]+1e-4);
  }
}

/*
 * Compute Batchnormalization based on pre-computed zeta: TODO
 */
void BatchnormalizationCMOZeta( float* InputTensor, int x, int y, float* beta, float* mean, float* zeta ) {
  assert( y==NUM_NEURONS );
  __m256 mean8ps;
  __m256 beta8ps;
  __m256 zeta8ps;

  for( int i=0; i<y; ++i ) {
    mean8ps = _mm256_set1_ps( mean[i] );
    beta8ps = _mm256_set1_ps( beta[i] );
    zeta8ps = _mm256_set1_ps( zeta[i] );
    for( int j=0; j<x; j+=8 ) {
      //InputTensor[i*x+j] = ((InputTensor[i*x+j] - mean[i])*zeta[i]) + beta[i];
      __m256 input = _mm256_load_ps( &InputTensor[i*x+j] );
      //__m256 result = _mm256_add_ps( _mm256_fmsub_ps( input, mean8ps, zeta8ps ), beta8ps );
      __m256 result = _mm256_fmadd_ps( _mm256_sub_ps( input, mean8ps ), zeta8ps, beta8ps );
      _mm256_storeu_ps( &InputTensor[i*x+j], result );
    }
  }
}

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

double find_maxima( float* Tensor, int x, int y ) {
  double max=0.0;
  for( int el=0; el<x*y; ++el ) {
    //if( max < abs(Tensor[el]) ) max = abs(Tensor[el]);
    if( (double) max < (double) sqrt((double)pow(Tensor[el],2)) ) max = (double)sqrt((double)pow(Tensor[el],2));
    //if( el<3 ) std::cout << "val=" << Tensor[el] << std::endl;
  }
  std::cout << "max=" << max << std::endl;
  return max;
}

void ternarize( float* WeightTensor, float Wp, float Wn, float threshold, int x, int y ) {
  double delta = /*0.427*/ 0.4 * find_maxima( WeightTensor, x, y );
  std::cout << "delta=" << delta << std::endl;
  float sparsity, zero_count=0.0;
  float *mask = (float*)malloc( x * y * sizeof(float) );
  for( int el=0; el<x*y; ++el ) {
    if( WeightTensor[el]>=delta ) {
      WeightTensor[el] = Wp;
      //mask[el]=Wp;
    }
    else if( WeightTensor[el]<=-delta ) {
      WeightTensor[el] = Wn;
      //mask[el]=-Wn;
    }
    else {
      WeightTensor[el] = 0.0;
      zero_count++;
    }
    //std::cout << "weight=" << WeightTensor[el] << std::endl;
  }
  /*for( int el=0; el<x*y; ++el ) {
    if( WeightTensor[el]>0 ) {
      WeightTensor[el]=+1*mask[el];
    }
    else if( WeightTensor[el]<0 ) {
      WeightTensor[el]=1*mask[el];
    }
    else {
      WeightTensor[el]=0.0;
    }
  }*/
  free(mask);
  sparsity=zero_count/(float)(x*y);
  std::cout << "sparstiy in layer=" << sparsity*100 << "% : " << zero_count << "/" << (x*y) << std::endl;
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
    cnpy::npz_t parameters_npz = cnpy::npz_load("../../trained_models/mnist-t40-final.npz");
    //cnpy::npz_t parameters_npz = cnpy::npz_load("../../trained_models/mnist-t20.npz");
    //cnpy::npz_t parameters_npz = cnpy::npz_load("./mnist-float32-nn.npz");

    assert( FRAME_SIZE == parameters_npz["fc0/W:0"].shape[0] );
    assert( NUM_NEURONS == parameters_npz["fc0/W:0"].shape[1] );
    float *WeightTensor0 = parameters_npz["fc0/W:0"].data<float>();
    //float *WeightTensor0 = (float*)malloc( NUM_NEURONS * FRAME_SIZE * sizeof(float) );
    //float Wp0 = parameters_npz["fc0/Wp:0"].data<float>();
    //std::cout << "dim=" << parameters_npz["fc0/Wp:0"].shape[0];
    //assert( Wp0[0] != 0.0 );
    //float *Wn0 = parameters_npz["fc0/Wn:0"].data<float>();
    //assert( Wn0[0] != 0.0 );
    float Wp0 = 0.7502421;
    float Wn0 = -1.2224674;
    ternarize( WeightTensor0, Wp0, Wn0, 0.10, FRAME_SIZE, NUM_NEURONS );
    float *WeightTensor0T = (float*)malloc( NUM_NEURONS * FRAME_SIZE * sizeof(float) );
    transposeMatrix( WeightTensor0, WeightTensor0T, FRAME_SIZE, NUM_NEURONS );
    uint16_t** SparseList0=NULL;
    createSparseList( WeightTensor0T, SparseList0, Wp0, Wn0, FRAME_SIZE, NUM_NEURONS);
    free( WeightTensor0T );
    assert( NUM_NEURONS == parameters_npz["fc0/b:0"].shape[0] );
    //float *bias0 = parameters_npz["fc0/b:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn0/beta:0"].shape[0] );
    float *beta0 = parameters_npz["bn0/beta:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn0/gamma:0"].shape[0] );
    float *gamma0 = parameters_npz["bn0/gamma:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn0/mean/EMA:0"].shape[0] );
    float *mean0 = parameters_npz["bn0/mean/EMA:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn0/variance/EMA:0"].shape[0] );
    float *variance0 = parameters_npz["bn0/variance/EMA:0"].data<float>();
    float *zeta0 = (float*)malloc( NUM_NEURONS * sizeof(float) );
    compute_zeta( zeta0, NUM_NEURONS, gamma0, variance0 );

    assert( NUM_NEURONS == parameters_npz["fc1/W:0"].shape[0] );
    assert( NUM_NEURONS == parameters_npz["fc1/W:0"].shape[1] );
    float *WeightTensor1 = parameters_npz["fc1/W:0"].data<float>();
    //float *WeightTensor1 = (float*)malloc( NUM_NEURONS * NUM_NEURONS * sizeof(float) );

    //float *Wp1 = parameters_npz["fc1/Wp:0"].data<float>();
    //assert( Wp1[0] != 0.0 );
    //float *Wn1 = parameters_npz["fc1/Wn:0"].data<float>();
    //assert( Wn1[0] != 0.0 );
    float Wp1 = 0.8800015;
    float Wn1 = -1.1117288;
    ternarize( WeightTensor1, Wp1, Wn1, 0.10, NUM_NEURONS, NUM_NEURONS );
    float *WeightTensor1T = (float*)malloc( NUM_NEURONS * NUM_NEURONS * sizeof(float) );
    transposeMatrix( WeightTensor1, WeightTensor1T, NUM_NEURONS, NUM_NEURONS );
    //uint16_t** SparseList1=NULL;
    //createSparseList( WeightTensor1T, SparseList1, Wp1, Wn1, NUM_NEURONS, NUM_NEURONS);
    //sparse_list_t *SparseListnew1=NULL;
    /*SparseListnew1 = (sparse_list_t*) malloc( NUM_NEURONS * sizeof(sparse_list_t) );
    if(SparseListnew1==NULL){
      std::cout << "[ERROR] Malloc1 failed!" << std::endl;
      return 1;
    }
    for( int i=0; i<NUM_NEURONS; ++i ) {
      SparseListnew1[i].list = (sparse_element_t*) malloc( NUM_NEURONS * sizeof(sparse_element_t));
      if(SparseListnew1[i].list==NULL){
        std::cout << "[ERROR] Malloc2 failed!" << std::endl;
        return 1;
      }
    }*/
    sparse_list_t *SparseListnew1=createSparseListv2( WeightTensor1T, Wp1, Wn1, NUM_NEURONS, NUM_NEURONS);
    free( WeightTensor1T );

    //createSparseList( WeightTensor1 )
    assert( NUM_NEURONS == parameters_npz["fc1/b:0"].shape[0] );
    //float *bias1 = parameters_npz["fc1/b:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn1/beta:0"].shape[0] );
    float *beta1 = parameters_npz["bn1/beta:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn1/gamma:0"].shape[0] );
    float *gamma1 = parameters_npz["bn1/gamma:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn1/mean/EMA:0"].shape[0] );
    float *mean1 = parameters_npz["bn1/mean/EMA:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn1/variance/EMA:0"].shape[0] );
    float *variance1 = parameters_npz["bn1/variance/EMA:0"].data<float>();
    float *zeta1 = (float*)malloc( NUM_NEURONS * sizeof(float) );
    compute_zeta( zeta1, NUM_NEURONS, gamma1, variance1 );

    assert( NUM_NEURONS == parameters_npz["fc2/W:0"].shape[0] );
    assert( NUM_NEURONS == parameters_npz["fc2/W:0"].shape[1] );
    float *WeightTensor2 = parameters_npz["fc2/W:0"].data<float>();
    //float *WeightTensor2 = (float*)malloc( NUM_NEURONS * NUM_NEURONS * sizeof(float) );
    //float *Wp2 =  parameters_npz["fc2/Wp:0"].data<float>();
    //assert( Wp2[0] != 0.0 );
    //float *Wn2 = parameters_npz["fc2/Wn:0"].data<float>();
    //assert( Wn2[0] != 0.0 );
    float Wp2 = 0.86801404;
    float Wn2 = -1.119899;
    ternarize( WeightTensor2, Wp2, Wn2, 0.10, NUM_NEURONS, NUM_NEURONS );
    float *WeightTensor2T = (float*)malloc( NUM_NEURONS * NUM_NEURONS * sizeof(float) );
    transposeMatrix( WeightTensor2, WeightTensor2T, NUM_NEURONS, NUM_NEURONS );
    //uint16_t** SparseList2=NULL;
    //createSparseList( WeightTensor2T, SparseList2, Wp2, Wn2, NUM_NEURONS, NUM_NEURONS);
    sparse_list_t *SparseList2 = createSparseListv2( WeightTensor2T, Wp2, Wn2, NUM_NEURONS, NUM_NEURONS);
    free( WeightTensor1T );
    assert( NUM_NEURONS == parameters_npz["fc2/b:0"].shape[0] );
    //float *bias2 = parameters_npz["fc2/b:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn2/beta:0"].shape[0] );
    float *beta2 = parameters_npz["bn2/beta:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn2/gamma:0"].shape[0] );
    float *gamma2 = parameters_npz["bn2/gamma:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn2/mean/EMA:0"].shape[0] );
    float *mean2 = parameters_npz["bn2/mean/EMA:0"].data<float>();
    assert( NUM_NEURONS == parameters_npz["bn2/variance/EMA:0"].shape[0] );
    float *variance2 = parameters_npz["bn2/variance/EMA:0"].data<float>();
    float *zeta2 = (float*)malloc( NUM_NEURONS * sizeof(float) );
    compute_zeta( zeta2, NUM_NEURONS, gamma2, variance2 );

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
    //float threshold = std::atof(argv[1]);
    //int bit_a = std::atoi(argv[2]);
    uint16_t *Input0Tint8 = (uint16_t*)malloc( BATCH_SIZE * FRAME_SIZE * sizeof(uint16_t) );
    float *Input0T = (float*)malloc( BATCH_SIZE * FRAME_SIZE * sizeof(float) );
    float *OutputTensor0T = (float*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(float) );
    uint16_t *OutputTensor0Tint8 = (uint16_t*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(uint16_t) );
    uint16_t *indices = (uint16_t*)malloc( BATCH_SIZE * (NUM_NEURONS+1) * sizeof(uint16_t) );
    //input_t *SAOutputTensor0T = (input_t*)malloc( BATCH_SIZE * (NUM_NEURONS+1) * sizeof(input_t) );
    //float *SAOutputTensor0T = (float*)malloc( 2* BATCH_SIZE * NUM_NEURONS) * sizeof(float) );
    float *OutputTensor1T = (float*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(float) );
    uint16_t *OutputTensor1Tint8 = (uint16_t*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(uint16_t) );
    //float *OutputTensor0T = (float*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(float) );
    float *OutputTensor2T = (float*)malloc( BATCH_SIZE * NUM_NEURONS * sizeof(float) );
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
      transposeMatrix( Input0, Input0Tint8, BATCH_SIZE, FRAME_SIZE, 8 );
      transposeMatrix( Input0, Input0T, BATCH_SIZE, FRAME_SIZE );
      //center_pixel_to_zero( Input0, BATCH_SIZE, FRAME_SIZE );

      start = std::chrono::system_clock::now();
      // Layer 0
      //FullyConnected( Input0, WeightTensor0, OutputTensor0, bias0, BATCH_SIZE, FRAME_SIZE, NUM_NEURONS );
      //FcBnReLUAVX2( Input0Tint8, SparseList0, OutputTensor0T, Wp0, Wn0, BATCH_SIZE, FRAME_SIZE, NUM_NEURONS, 8 );
      //SparseDotProduct( Input0T, SparseList0, OutputTensor0T, Wp0, Wn0, BATCH_SIZE, FRAME_SIZE, NUM_NEURONS );
      SparseDotProductAVX2( Input0T, SparseList0, OutputTensor0T, Wp0, Wn0, BATCH_SIZE, FRAME_SIZE, NUM_NEURONS );
      //SparseDotProductnew( Input0T, SparseList0, OutputTensor0T, Wp0, Wn0, BATCH_SIZE, FRAME_SIZE, NUM_NEURONS );

      //transposeMatrix( OutputTensor0T, OutputTensor0, NUM_NEURONS, BATCH_SIZE );
      //BatchnormalizationCMO( OutputTensor0T, BATCH_SIZE, NUM_NEURONS, beta0, gamma0, mean0, variance0 );
      BatchnormalizationCMOZeta( OutputTensor0T, BATCH_SIZE, NUM_NEURONS, beta0, mean0, zeta0 );
      zero_count += (unsigned long long int) ReLU( OutputTensor0T, BATCH_SIZE, NUM_NEURONS, 0.0 /*0.25*/, 8 );
      activation_count += (unsigned long long int) BATCH_SIZE * NUM_NEURONS;

      // Layer 1
      //FullyConnected( OutputTensor0, WeightTensor1, OutputTensor1, bias1, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS );
      //transposeMatrix( OutputTensor0, OutputTensor0T, BATCH_SIZE, NUM_NEURONS );
      //SAFcBnReLUAVX2( OutputTensor0Tint8, indices, SparseList1, OutputTensor1T, Wp1, Wn1, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS, 8 );
      //SparseDotProductAVX2( OutputTensor0T, SparseList1, OutputTensor1T, Wp1, Wn1, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS );
      SparseDotProductnew( OutputTensor0T, SparseListnew1, OutputTensor1T, Wp1, Wn1, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS );
      //SASparseDotProduct( SAOutputTensor0T, indices, SparseList1, OutputTensor1T, Wp1, Wn1, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS );
      //transposeMatrix( OutputTensor1T, OutputTensor1, NUM_NEURONS, BATCH_SIZE );
      BatchnormalizationCMOZeta( OutputTensor1T, BATCH_SIZE, NUM_NEURONS, beta1, mean1, zeta1 );
      //BatchnormalizationCMO( OutputTensor1T, BATCH_SIZE, NUM_NEURONS, beta1, gamma1, mean1, variance1 );
      zero_count += (unsigned long long int) ReLU( OutputTensor1T, OutputTensor1Tint8, BATCH_SIZE, NUM_NEURONS, 0.0 /*0.32*/, 8 );
      activation_count += (unsigned long long int) BATCH_SIZE * NUM_NEURONS;

      // Layer 2
      //FullyConnected( OutputTensor1, WeightTensor2, OutputTensor2, bias2, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS );
      //transposeMatrix( OutputTensor1, OutputTensor1T, BATCH_SIZE, NUM_NEURONS );
      SparseDotProductnew( OutputTensor1T, SparseList2, OutputTensor2T, Wp2, Wn2, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS );
      //FcBnReLUAVX2( OutputTensor1Tint8, SparseList2, OutputTensor2T, Wp2, Wn2, BATCH_SIZE, NUM_NEURONS, NUM_NEURONS, 8 );
      //transposeMatrix( OutputTensor2T, OutputTensor2, NUM_NEURONS, BATCH_SIZE );
      BatchnormalizationCMOZeta( OutputTensor2T, BATCH_SIZE, NUM_NEURONS, beta2, mean2, zeta2 );
      //BatchnormalizationCMO( OutputTensor2T, BATCH_SIZE, NUM_NEURONS, beta2, gamma2, mean2, variance2 );
      zero_count += (unsigned long long int) ReLU( OutputTensor2T, BATCH_SIZE, NUM_NEURONS, 0.0, 32 );
      activation_count += (unsigned long long int) BATCH_SIZE * NUM_NEURONS;

      transposeMatrix( OutputTensor2T, OutputTensor2, NUM_NEURONS, BATCH_SIZE );
      // Layer 3
      FullyConnected( OutputTensor2, WeightTensor3, logits, bias3, BATCH_SIZE, NUM_NEURONS, NUM_UNITS );
      Softmax( logits, BATCH_SIZE, NUM_UNITS );

      end = std::chrono::system_clock::now();
      elapsed_seconds = end-start;
      overall_time += elapsed_seconds.count();

      //transposeMatrix( OutputTensor3T, OutputTensor3, NUM_NEURONS, BATCH_SIZE );
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
    free(Input0Tint8);
    free(OutputTensor0T);
    free(OutputTensor1T);
    free(OutputTensor2T);

    free(Input0);
    free(OutputTensor0);
    free(OutputTensor1);
    free(OutputTensor2);
    free(logits);

    //free(SparseListnew1);

    free(zeta0); free(zeta1); free(zeta2);

    return 0;
}
