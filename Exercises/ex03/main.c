/*	Advanced Parallel Computing - Exercise 3

	Jona Neef
	Nikolas Krätzschmar
	Philipp Walz

	usage: ./main c n
		c: number of increments (optional, default 1024*1024)
		n: number of threads (optional, default number of cpu cores)

 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/sysinfo.h>

typedef unsigned int uint;

typedef struct
{
	uint* cntr;
	const uint c;
	pthread_mutex_t* mtx;
	bool* lock;
} args_t;

void* inc(void* args);
void* inc_mtx(void* args);
void* inc_atomic(void* args);
void* inc_lock(void* args);

void lock_rmw(bool* lock)
{
	while(!__sync_bool_compare_and_swap(lock, 0, 1));
}

void unlock_rmw(bool* lock)
{
	__sync_bool_compare_and_swap(lock, 1, 0);
}

int main(int argc, char** argv)
{
	const uint c = argc >= 2 ? atoi(argv[1]) : 1024 * 1024;
	const uint n = argc >= 3 ? atoi(argv[2]) : get_nprocs();

	if(c % n) exit(-1);

	uint cntr, ci = c / n;

	pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
	bool lock = 0;

	args_t args = { &cntr, ci, &mtx, &lock };

	{ // naive
		cntr = 0;
		pthread_t t[n];

		for(uint i = 0; i < n; ++i) if(pthread_create(&t[i], NULL, inc, &args)) exit(-1);
		for(uint i = 0; i < n; ++i) if(pthread_join(t[i], NULL)) exit(-1);

		printf("naive    %010d %c= %010d\n", cntr, cntr == c ? '=' : '!', c);
	}

	{ // mutex
		cntr = 0;
		pthread_t t[n];

		for(uint i = 0; i < n; ++i) if(pthread_create(&t[i], NULL, inc_mtx, &args)) exit(-1);
		for(uint i = 0; i < n; ++i) if(pthread_join(t[i], NULL)) exit(-1);

		printf("mutex    %010d %c= %010d\n", cntr, cntr == c ? '=' : '!', c);
	}

	{ // atomics
		cntr = 0;
		pthread_t t[n];

		for(uint i = 0; i < n; ++i) if(pthread_create(&t[i], NULL, inc_atomic, &args)) exit(-1);
		for(uint i = 0; i < n; ++i) if(pthread_join(t[i], NULL)) exit(-1);

		printf("atomics  %010d %c= %010d\n", cntr, cntr == c ? '=' : '!', c);
	}

	{ // lock_rmw
		cntr = 0;
		pthread_t t[n];

		for(uint i = 0; i < n; ++i) if(pthread_create(&t[i], NULL, inc_mtx, &args)) exit(-1);
		for(uint i = 0; i < n; ++i) if(pthread_join(t[i], NULL)) exit(-1);

		printf("lock_rmw %010d %c= %010d\n", cntr, cntr == c ? '=' : '!', c);
	}

	return 0;
}

void* inc(void* args)
{
	uint* cntr = ((args_t*) args)->cntr;
	const uint c = ((args_t*) args)->c;

	for(uint i = 0; i < c; ++i) ++(*cntr);

	return NULL;
}

void* inc_mtx(void* args)
{
	uint* cntr = ((args_t*) args)->cntr;
	const uint c = ((args_t*) args)->c;
	pthread_mutex_t* mtx = ((args_t*) args)->mtx;

	for(uint i = 0; i < c; ++i)
	{
		pthread_mutex_lock(mtx);
		++(*cntr);
		pthread_mutex_unlock(mtx);
	}

	return NULL;
}

void* inc_atomic(void* args)
{
	uint* cntr = ((args_t*) args)->cntr;
	const uint c = ((args_t*) args)->c;

	for(uint i = 0; i < c; ++i) __sync_add_and_fetch(cntr, 1);

	return NULL;
}

void* inc_lock(void* args)
{
	uint* cntr = ((args_t*) args)->cntr;
	const uint c = ((args_t*) args)->c;
	bool* lock = ((args_t*) args)->lock;

	for(uint i = 0; i < c; ++i)
	{
		lock_rmw(lock);
		++(*cntr);
		unlock_rmw(lock);
	}

	return NULL;
}
