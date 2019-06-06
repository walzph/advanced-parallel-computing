/*	Advanced Parallel Computing - Exercise 5

	Jona Neef
	Nikolas Krätzschmar
	Philipp Walz

	usage: ./main <0, 1>
        0 - pthread
        1 - global counter

 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>

#define NUM_BARRIER_STEPS 500
typedef unsigned int uint;

pthread_barrier_t pthread_barrier;
uint global_pssd_barriers;

typedef struct
{
	uint* short_id;
	uint* pssd_barriers_cnt;
} args_t;

void* thread_fn_pthread(void *args) 
{
    uint* thread_id = ((args_t*) args)->short_id;
	uint* pssd_barriers_cnt = ((args_t*) args)->pssd_barriers_cnt;

    uint microseconds = 1;
	if (*thread_id == 0) microseconds = 5;   

	for (int k=0; k<NUM_BARRIER_STEPS; ++k) 
	{
		usleep(microseconds);
		pthread_barrier_wait(&pthread_barrier);
        (*pssd_barriers_cnt)++;
	}
	return NULL;
}

void* thread_fn_counter (void *args)
{
    uint* thread_id = ((args_t*) args)->short_id;
    uint* pssd_barriers_cnt = ((args_t*) args)->pssd_barriers_cnt;

    uint microseconds = 1;
    if (*thread_id == 0) microseconds = 5;   

	for (uint k=1; k<NUM_BARRIER_STEPS; ++k) 
	{
		usleep(microseconds);
        if (*thread_id == 0) {
            global_pssd_barriers++;
        }     
        while ( global_pssd_barriers < k); //spin wait
        (*pssd_barriers_cnt)++;
	}
    return NULL;
}

int main( int argc, char * argv[] )
{
	/* parse arguments */
	const uint f = argc >= 2 ? atoi(argv[1]) : 0;
	const uint n = argc >= 3 ? atoi(argv[2]) : get_nprocs();
    
    pthread_t t[n];
    uint short_ids[n];
	uint pssd_barriers_cnt[n];

    pthread_barrier_init(&pthread_barrier, NULL, n);

    for(uint i = 0; i < n; ++i) 
    {
        short_ids[i] = i;
        pssd_barriers_cnt[i] = 0;
        args_t args = { &short_ids[i] , &pssd_barriers_cnt[i] };
        switch (f) {
            case 1: if(pthread_create(&t[i], NULL, thread_fn_pthread, &args)) exit(2); break;
            case 0: if(pthread_create(&t[i], NULL, thread_fn_counter, &args)) exit(2); break;
        }
    }
    printf("\n Created threads \n");

    for(uint i = 0; i < n; ++i) if(pthread_join(t[i], NULL)) exit(3);
    
    for (uint i=0; i<sizeof(pssd_barriers_cnt)/sizeof(uint); i++ ) {
        printf("thread %d: passed %d barriers.\n", i, pssd_barriers_cnt[i]);
    }
    pthread_barrier_destroy(&pthread_barrier);
}