.PHONY: build clean rebuild

build: main

rebuild: clean build

clean:
	rm -f main

main: main.c
	$(CC) -std=c99 -g -Wall -pthread $< -o $@
