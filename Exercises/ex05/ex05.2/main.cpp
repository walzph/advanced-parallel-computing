/*	Advanced Parallel Computing - Exercise 5

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
#include <unistd.h>

#define NUM_VERSIONS 1
#define NUM_BARRIER_STEPS 500

typedef unsigned int uint;

typedef struct
{
	uint* short_id;
	uint* pssd_barriers_cnt;
	uint* barrier;
} args_t;

void* threadFn(void* args);

/* 
 PTHREAD barrier
*/
void init(pthread_barrier_t *barrier, int num_threads) {
	pthread_barrier_init(barrier, NULL, num_threads);
}

void wait(pthread_barrier_t *barrier) { 
	pthread_barrier_wait(barrier);
}

void destroy(pthread_barrier_t *barrier) {
	pthread_barrier_destroy(barrier);
}

/* 
 Counter barrier
*/
typedef struct
{
	uint* test;
} counter_barrier_t;

void init(counter_barrier_t *barrier, int num_threads) {
	return;
}

void wait(counter_barrier_t *barrier) { 
	return;
}

void destroy(counter_barrier_t *barrier) {
	return;
}


// void benchmark();
// void print_benchmark_header();

template<typename T>
void* threadFn(void *args) {
	uint* thread_id = ((args_t*) args)->short_id;
	uint* pssd_barriers_cnt = ((args_t*) args)->pssd_barriers_cnt;
	T* mybarrier = (T*)(((args_t*) args)->barrier);

	int wait_sec = 0.00001;
	if (*thread_id == 0) wait_sec = 0.00005;

	// perform k barrier steps
	for (int k=0; k<NUM_BARRIER_STEPS; ++k) 
	{
		printf("thread %d: Wait for %d seconds.\n", (int)(*thread_id), wait_sec);
		sleep(wait_sec);
		printf("thread %d: I'm ready...\n", (int)(*thread_id));

		wait(mybarrier);

		(*pssd_barriers_cnt)++;
		printf("thread %d: passed barrier no. %d. going on!\n", (int)(*thread_id), (int)(*pssd_barriers_cnt));
	}
	return NULL;
}


int main( int argc, char * argv[] )
{
	/* parse arguments */
	const uint c = argc >= 2 ? atoi(argv[1]) : 3 * 1024 * 1024;
	const uint n = argc >= 3 ? atoi(argv[2]) : get_nprocs();
	// const bool b = argc >= 4 ? (*argv[3] == 'y' || *argv[3] == 't' || *argv[3] == '1') : 1;

	if(c % n) exit(1);

	pthread_t t[n];
	uint short_ids[n];
	uint pssd_barriers_cnt[n];

	uint* barriers[1];
	pthread_barrier_t pthread_barrier; init(&pthread_barrier, n); barriers[0] = (uint*) &pthread_barrier;
	// counter_barrier_t counter_barrier; init(&pthread_barrier, n); barriers[1] = (uint*) &counter_barrier;
	// ...

	// test all barrier implementations
	for(int j = 0; j<NUM_VERSIONS; j++)
	{	
		// span n threads
		for(uint i = 0; i < n; ++i) 
		{
			short_ids[i] = i;
			pssd_barriers_cnt[i] = 0;
			args_t args = { &short_ids[i] , &pssd_barriers_cnt[i] , barriers[j] };
			if(pthread_create(&t[i], NULL, threadFn<pthread_barrier_t>, &args)) exit(2);
		}
		
		// printf("main() is ready.\n");
		// wait(&pthread_barrier);
		
		for(uint i = 0; i < n; ++i) 
		{
			if(pthread_join(t[i], NULL)) exit(3);
		}

		destroy(&pthread_barrier);

		// printf("%8s %010d %c= %010d\n", func->s, cntr, cntr == c ? '=' : '!', c);
	}

	// if(b) benchmark();

	return 0;
}


// void benchmark()
// {
// 	const uint c = 3 * 5 * (128 * 1024);
// 	const uint ns[] = { 1, 2, 4, 8, 12, 16, 24, 32, 40, 48 };
// 	const uint ns_len = sizeof(ns) / sizeof(uint);

// 	pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
// 	bool lock = 0;
// 	mcs_t* mcs_lock = NULL;

// 	struct timespec t0, t1;

// 	print_benchmark_header();

// 	for(const uint* n = ns; n < (ns + ns_len); ++n)
// 	{
// 		printf("%9d", *n);

// 		uint cntr, ci = c / *n;
// 		args_t args = { &cntr, ci, &mtx, &lock, &mcs_lock };

// 		for(const func_def* func = (funcs + 1); func < (funcs + funcs_len); ++func)
// 		{
// 			cntr = 0;
// 			pthread_t t[*n];

// 			clock_gettime(CLOCK_MONOTONIC, &t0);
// 			for(uint i = 0; i < *n; ++i) if(pthread_create(&t[i], NULL, func->f, &args)) exit(2);
// 			for(uint i = 0; i < *n; ++i) if(pthread_join(t[i], NULL)) exit(3);
// 			clock_gettime(CLOCK_MONOTONIC, &t1);

// 			double time = (double) (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
// 			double rate = c / time;

// 			printf(" %20f %20.2f", time, rate);
// 		}
// 		printf("\n");
// 		fflush(stdout);
// 	}
// }

// void print_benchmark_header()
// {
// 	printf("\n\nbenchmark");
// 	for(const func_def* func = (funcs + 1); func < (funcs + funcs_len); ++func) printf(" %20s %20s", func->s, "");
// 	printf("\nn threads");
// 	for(uint i = 1; i < funcs_len; ++i) printf(" %20s %20s", "exec time (s)", "updates / sec");
// 	printf("\n");
// 	fflush(stdout);
// }
