.PHONY: build clean rebuild

build: main

rebuild: clean build

clean:
	rm -f main

main: main.cpp sparse.h
	$(CXX) -O3 -g -mavx2 -mfma -Imnist/include -lcnpy $< -o $@
