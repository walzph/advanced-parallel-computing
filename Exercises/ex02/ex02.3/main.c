#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include "chTimer.h"
void chTimerGetTime(chTimerTimestamp *p);
double chTimerElapsedTime(chTimerTimestamp *pStart, chTimerTimestamp *pStop);

typedef unsigned int uint;

static const uint n = 1024*1024;

typedef struct
{
	char* src;
	uint n;
} args_t;

void *load(void* args);

int main()
{
	for(uint t = 1; t <= 1<<6; t <<= 1)
	{
		uint nt = n / t;
		char (*buf)[nt] = malloc(n);

		pthread_t threads[t];

		chTimerTimestamp t0, t1;

		chTimerGetTime(&t0);

		for(uint i = 0; i < t; ++i)
		{
			args_t* args = malloc(sizeof(args_t));
			args->src = (char*) (buf++);
			args->n = nt;

			if(pthread_create(&threads[i], NULL, load, args)) exit(-1);
		}

		for(uint i = 0; i < t; ++i) if(pthread_join(threads[i], NULL)) exit(-1);

		chTimerGetTime(&t1);

		free(buf-t);

		double ms = n / (1e9 * chTimerElapsedTime(&t0, &t1));
		printf("%d, %f\n", t, ms);
	}
	
	return 0;
}

void *load(void* args)
{
	volatile char* src = ((args_t*) args)->src;
	uint nt = ((args_t*) args)->n;
	free(args);

	volatile register char c;
	for(uint i = 0; i < nt; ++i) c = *src++;

	return NULL;
}
