.PHONY: build rebuild clean

CFLAGS+=-O3 -I./inc/ -I./third_party/mnist/include
LDFLAGS+=-lcnpy

build: main

rebuild: clean build

clean:
	rm -f main

main: benchmarks/mnist/main-dc.cpp inc/sparse_ops.h
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $<
