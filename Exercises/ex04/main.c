/*	Advanced Parallel Computing - Exercise 3

	Jona Neef
	Nikolas Krätzschmar
	Philipp Walz

	usage: ./main c n b
		c: number of increments (optional, default 3*1024*1024)
		n: number of threads (optional, default number of cpu cores)
		b: run benchmark (yes/no) (optional, default yes)

 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <time.h>

typedef unsigned int uint;

typedef struct mcs_t
{
	struct mcs_t* next;
	bool locked;
} mcs_t;

typedef struct
{
	uint* cntr;
	const uint c;
	pthread_mutex_t* mtx;
	bool* lock;
	mcs_t** mcs_lock;
} args_t;

void* inc(void* args);
void* inc_mtx(void* args);
void* inc_atomic(void* args);
void* inc_lock(void* args);
void* inc_mcs(void* args);

/* custom locking mechanism */
void lock_rmw(bool* lock)
{
	while(!__sync_bool_compare_and_swap(lock, 0, 1));
}

void unlock_rmw(bool* lock)
{
	__sync_bool_compare_and_swap(lock, 1, 0);
}

/* MCS list based queing lock */
void lock_mcs(mcs_t** lock, mcs_t* node)
{
	node->next = NULL;
	mcs_t* prev = __atomic_exchange_n(lock, node, __ATOMIC_RELAXED);
	if(prev != NULL)
	{
		node->locked = 1;
		prev->next = node;
		while(node->locked);
	}
}

void unlock_mcs(mcs_t** lock, mcs_t* node)
{
	if(node->next == NULL)
	{
		mcs_t* _node = node;
		if (__atomic_compare_exchange_n(lock, &_node, NULL, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
		else while(node->next == NULL);
	}
	node->next->locked = 0;
}

void benchmark();

typedef struct
{
	void* (*f)(void*);
	const char* s;
} func_def;

/* functions to test / benchmark */
static const func_def funcs[] = {
	{ &inc,        "naive"    },
	{ &inc_mtx,    "mutex"    },
	{ &inc_atomic, "atomics"  },
	{ &inc_lock,   "lock_rmw" },
	{ &inc_mcs,    "lock_mcs" }
};
static const uint funcs_len = sizeof(funcs) / sizeof(func_def);

int main(int argc, char** argv)
{
	/* parse arguments */
	const uint c = argc >= 2 ? atoi(argv[1]) : 3 * 1024 * 1024;
	const uint n = argc >= 3 ? atoi(argv[2]) : get_nprocs();
	const bool b = argc >= 4 ? (*argv[3] == 'y' || *argv[3] == 't' || *argv[3] == '1') : 1;

	if(c % n) exit(1);

	uint cntr, ci = c / n;

	pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
	bool lock = 0;
	mcs_t* mcs_lock = NULL;

	args_t args = { &cntr, ci, &mtx, &lock, &mcs_lock };

	/* test all functions */
	for(const func_def* func = funcs; func < (funcs + funcs_len); ++func)
	{
		cntr = 0;
		pthread_t t[n];

		for(uint i = 0; i < n; ++i) if(pthread_create(&t[i], NULL, func->f, &args)) exit(2);
		for(uint i = 0; i < n; ++i) if(pthread_join(t[i], NULL)) exit(3);

		printf("%8s %010d %c= %010d\n", func->s, cntr, cntr == c ? '=' : '!', c);
	}

	if(b) benchmark();

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

void* inc_mcs(void* args)
{
	uint* cntr = ((args_t*) args)->cntr;
	const uint c = ((args_t*) args)->c;
	mcs_t** lock = ((args_t*) args)->mcs_lock;

	mcs_t node;

	for(uint i = 0; i < c; ++i)
	{
		lock_mcs(lock, &node);
		++(*cntr);
		unlock_mcs(lock, &node);
	}

	return NULL;
}

void print_benchmark_header();

void benchmark()
{
	const uint c = 3 * 5 * (128 * 1024);
	const uint ns[] = { 1, 2, 4, 8, 12, 16, 24, 32, 40, 48 };
	const uint ns_len = sizeof(ns) / sizeof(uint);

	pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
	bool lock = 0;
	mcs_t* mcs_lock = NULL;

	struct timespec t0, t1;

	print_benchmark_header();

	for(const uint* n = ns; n < (ns + ns_len); ++n)
	{
		printf("%9d", *n);

		uint cntr, ci = c / *n;
		args_t args = { &cntr, ci, &mtx, &lock, &mcs_lock };

		for(const func_def* func = (funcs + 1); func < (funcs + funcs_len); ++func)
		{
			cntr = 0;
			pthread_t t[*n];

			clock_gettime(CLOCK_MONOTONIC, &t0);
			for(uint i = 0; i < *n; ++i) if(pthread_create(&t[i], NULL, func->f, &args)) exit(2);
			for(uint i = 0; i < *n; ++i) if(pthread_join(t[i], NULL)) exit(3);
			clock_gettime(CLOCK_MONOTONIC, &t1);

			double time = (double) (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
			double rate = c / time;

			printf(" %20f %20.2f", time, rate);
		}
		printf("\n");
		fflush(stdout);
	}
}

void print_benchmark_header()
{
	printf("\n\nbenchmark");
	for(const func_def* func = (funcs + 1); func < (funcs + funcs_len); ++func) printf(" %20s %20s", func->s, "");
	printf("\nn threads");
	for(uint i = 1; i < funcs_len; ++i) printf(" %20s %20s", "exec time (s)", "updates / sec");
	printf("\n");
	fflush(stdout);
}
