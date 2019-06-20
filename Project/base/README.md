# Install dependencies

Create build directory and install CNPY
```shell
mkdir build
```
```shell
cd build
```
```shell
cmake ../third_party/cnpy/
```
```shell
make
```

# Run MLP on MNIST
```shell
cd benchmarks/mnist/
```
```shell
g++ -o mnist main.cpp -L../../build/ -lcnpy --std=c++11 -Wall -O3
```
```shell
./mnist
```
