.PHONY: build clean rebuild test

CFLAGS+=-O3 --std=c++14 -Imnist/include
LDFLAGS+=-lcnpy

build: main main_avx test

rebuild: clean build

clean:
	rm -f main main_avx test mat.o sparse.o

run: rebuild
	./main_avx
	./main
	./test

main_avx: main.cpp sparse_ops.h
	$(CXX) $(CFLAGS) $(LDFLAGS) -mavx2 -mfma -o $@ $<

main: main.cpp mat.o sparse.o
	$(CXX) $(CFLAGS) $(LDFLAGS) -D__NEW_SPARSE__ -o $@ $^

test: test.cpp mat.o sparse.o
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.cpp %.hpp
	$(CXX) $(CFLAGS) -c $<
