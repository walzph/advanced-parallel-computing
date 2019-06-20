
#include "../third_party/Eigen/Eigen/Dense"
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrixXf;

void DenseDotProduct( float* InputTensor, float* WeightTensor, float* Output, int m, int n, int k ) {
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
