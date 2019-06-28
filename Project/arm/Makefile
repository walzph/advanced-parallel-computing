.PHONY: build clean rebuild

build: main

rebuild: clean build

clean:
	rm -f main.o main

main.o: main.cpp sparse_ops.h sparse.hpp
	$(CXX) $(CFLAGS) -O3 --std=c++14 -mavx2 -mfma -Imnist/include -lcnpy -c $<

main: main.o
	$(CXX) $(LDFLAGS) -lcnpy $^ -o $@
