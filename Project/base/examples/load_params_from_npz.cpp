#include "../third_party/cnpy/cnpy.h"

/*
 *  This example simply loads parameters of a trained MLP with 1024 neurons
 *  and batchnormalization from a .npz file
 *
 *  Compile: g++ -o load_params_from_npz load_params_from_npz.cpp -L../build/ -lcnpy --std=c++11
 *
 *  The structure is as follows:
 *
 *   ['bn0/beta:0',
 *    'bn0/gamma:0',
 *    'bn0/mean/EMA:0',
 *    'bn0/variance/EMA:0',
 *    'bn1/beta:0',
 *    'bn1/gamma:0',
 *    'bn1/mean/EMA:0',
 *    'bn1/variance/EMA:0',
 *    'bn2/beta:0',
 *    'bn2/gamma:0',
 *    'bn2/mean/EMA:0',
 *    'bn2/variance/EMA:0',
 *    'fc0/W:0',
 *    'fc0/b:0',
 *    'fc1/W:0',
 *    'fc1/b:0',
 *    'fc2/W:0',
 *    'fc2/b:0',
 *    'fc3/W:0',
 *    'fc3/b:0']
 *
*/

int main() {

  std::cout << "Helo Numpy!" << std::endl;

  //cnpy::NpyArray arr2 = cnpy::npz_load("../trained_models/mnist-float32.npz","fc0/W:0");

  cnpy::npz_t my_npz = cnpy::npz_load("../trained_models/mnist-float32.npz");

  /*
   *    Layer 0
   */
  cnpy::NpyArray fc0_W = my_npz["fc0/W:0"];
  std::cout << "fc0/W:0 dims = " << fc0_W.shape.size() << std::endl;
  std::cout << "fc0/W:0 dim 0 = " << fc0_W.shape[0] << std::endl;
  std::cout << "fc0/W:0 dim 1 = " << fc0_W.shape[1] << std::endl;

  cnpy::NpyArray bn0_beta = my_npz["bn0/beta:0"];
  std::cout << "bn0/beta:0 dims = " << bn0_beta.shape.size() << std::endl;
  std::cout << "bn0/beta:0 dim 0 = " << bn0_beta.shape[0] << std::endl;
  cnpy::NpyArray bn0_gamma = my_npz["bn0/gamma:0"];
  std::cout << "bn0/gamma:0 dims = " << bn0_gamma.shape.size() << std::endl;
  std::cout << "bn0/gamma:0 dim 0 = " << bn0_gamma.shape[0] << std::endl;
  cnpy::NpyArray bn0_mean = my_npz["bn0/mean/EMA:0"];
  std::cout << "bn0/mean/EMA:0 dims = " << bn0_mean.shape.size() << std::endl;
  std::cout << "bn0/mean/EMA:0 dim 0 = " << bn0_mean.shape[0] << std::endl;
  cnpy::NpyArray bn0_variance = my_npz["bn0/variance/EMA:0"];
  std::cout << "bn0/variance/EMA:0 dims = " << bn0_variance.shape.size() << std::endl;
  std::cout << "bn0/variance/EMA:0 dim 0 = " << bn0_variance.shape[0] << std::endl;

  /*
   *    Layer 1
   */
  cnpy::NpyArray fc1_W = my_npz["fc1/W:0"];
  std::cout << "fc1/W:0 dims = " << fc1_W.shape.size() << std::endl;
  std::cout << "fc1/W:0 dim 0 = " << fc1_W.shape[0] << std::endl;
  std::cout << "fc1/W:0 dim 1 = " << fc1_W.shape[1] << std::endl;

  cnpy::NpyArray bn1_beta = my_npz["bn1/beta:0"];
  std::cout << "bn1/beta:0 dims = " << bn1_beta.shape.size() << std::endl;
  std::cout << "bn1/beta:0 dim 0 = " << bn1_beta.shape[0] << std::endl;
  cnpy::NpyArray bn1_gamma = my_npz["bn1/gamma:0"];
  std::cout << "bn1/gamma:0 dims = " << bn1_gamma.shape.size() << std::endl;
  std::cout << "bn1/gamma:0 dim 0 = " << bn1_gamma.shape[0] << std::endl;
  cnpy::NpyArray bn1_mean = my_npz["bn1/mean/EMA:0"];
  std::cout << "bn1/mean/EMA:0 dims = " << bn1_mean.shape.size() << std::endl;
  std::cout << "bn1/mean/EMA:0 dim 0 = " << bn1_mean.shape[0] << std::endl;
  cnpy::NpyArray bn1_variance = my_npz["bn1/variance/EMA:0"];
  std::cout << "bn1/variance/EMA:0 dims = " << bn1_variance.shape.size() << std::endl;
  std::cout << "bn1/variance/EMA:0 dim 0 = " << bn1_variance.shape[0] << std::endl;

  /*
   *    Layer 2
   */
  cnpy::NpyArray fc2_W = my_npz["fc2/W:0"];
  std::cout << "fc2/W:0 dims = " << fc2_W.shape.size() << std::endl;
  std::cout << "fc2/W:0 dim 0 = " << fc2_W.shape[0] << std::endl;
  std::cout << "fc2/W:0 dim 1 = " << fc2_W.shape[1] << std::endl;

  cnpy::NpyArray bn2_beta = my_npz["bn2/beta:0"];
  std::cout << "bn2/beta:0 dims = " << bn2_beta.shape.size() << std::endl;
  std::cout << "bn2/beta:0 dim 0 = " << bn2_beta.shape[0] << std::endl;
  cnpy::NpyArray bn2_gamma = my_npz["bn2/gamma:0"];
  std::cout << "bn2/gamma:0 dims = " << bn2_gamma.shape.size() << std::endl;
  std::cout << "bn2/gamma:0 dim 0 = " << bn2_gamma.shape[0] << std::endl;
  cnpy::NpyArray bn2_mean = my_npz["bn2/mean/EMA:0"];
  std::cout << "bn2/mean/EMA:0 dims = " << bn2_mean.shape.size() << std::endl;
  std::cout << "bn2/mean/EMA:0 dim 0 = " << bn2_mean.shape[0] << std::endl;
  cnpy::NpyArray bn2_variance = my_npz["bn2/variance/EMA:0"];
  std::cout << "bn2/variance/EMA:0 dims = " << bn2_variance.shape.size() << std::endl;
  std::cout << "bn2/variance/EMA:0 dim 0 = " << bn2_variance.shape[0] << std::endl;

  /*
   *    Layer 3
   */
  cnpy::NpyArray fc3_W = my_npz["fc3/W:0"];
  std::cout << "fc3/W:0 dims = " << fc3_W.shape.size() << std::endl;
  std::cout << "fc3/W:0 dim 0 = " << fc3_W.shape[0] << std::endl;
  std::cout << "fc3/W:0 dim 1 = " << fc3_W.shape[1] << std::endl;

  return 0;
}
