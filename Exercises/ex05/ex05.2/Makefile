.PHONY: build clean rebuild

build: main

rebuild: clean build

clean:
	rm -f main

main: main.cpp
	g++ -O0 -g -pthread -DDEBUG -Wall -Wextra $^ -o $@ -I./inc/
