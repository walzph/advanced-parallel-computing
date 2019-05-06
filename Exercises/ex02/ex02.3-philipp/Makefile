.PHONY: build clean rebuild

build: main


clean:
	rm -f main

rebuild: clean build

main: src/main.cpp
	g++ -O0 -g -pthread -DDEBUG -Wall -Wextra $^ -o $@ -I./inc/
