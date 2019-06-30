.PHONY: build clean rebuild test

build: main main_avx

rebuild: clean build

clean:
	rm -f main main_avx

run: rebuild
	./main_avx
	./main

test: test.cpp sparse.hpp
		$(CXX) $(CFLAGS) $(LDFLAGS) -O3 --std=c++14 -Imnist/include -lcnpy -o $@ $<

main_avx: main.cpp sparse_ops.h
	$(CXX) $(CFLAGS) $(LDFLAGS) -O3 --std=c++14 -mavx2 -mfma -Imnist/include -lcnpy -o $@ $<

main: main.cpp sparse.hpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -D__NEW_SPARSE__ -O3 --std=c++14 -Imnist/include -lcnpy -o $@ $<
