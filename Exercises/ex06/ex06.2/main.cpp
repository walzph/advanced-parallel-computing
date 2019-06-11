/*	Advanced Parallel Computing - Exercise 6

	Jona Neef
	Nikolas Krätzschmar
	Philipp Walz

	usage: ./main <array_size> <num_threads> 

 */

// #include <stdlib.h>
// #include <stdio.h>
// #include <stdbool.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <iostream> 
#include <iterator> 
#include <map> 
#include <vector>
#include <bits/stdc++.h> 

#define THREAD_CNT 4
#define MIN 0 
#define MAX 100
#define DEBUGGING false
#define DEFAULT_VALS false

using namespace std;
typedef unsigned int uint;

vector<int> v;
int full_sum;
int thread_cnt;
map<int, int> ids;
pthread_barrier_t barrier;

typedef struct
{
	uint* arg;
} args_t;

void printMap(map<int, int> m) {
    map<int, int>::iterator itr; 
    cout << "\nThe map is : \n";
    cout << "\tKEY\tELEMENT\n"; 
    for (itr = m.begin(); itr != m.end(); ++itr) { 
        cout << '\t' << itr->first 
            << '\t' << itr->second << '\n'; 
    } 
    cout << endl; 
}

vector<int> preScan2Scan(vector<int> v, int s) {
    v.erase(v.begin());
    v.push_back(s);    
    return v;
}

void printVector(vector<int> v) {
    for (vector<int>::const_iterator i = v.begin(); i != v.end(); ++i)
    cout << *i << ' ';
}

void* thread_fn_pthread(void *args) 
{
    usleep(1);
    int tid = (int) pthread_self();
    int index = ids.find(tid)->second;
    uint* arg = ((args_t*) args)->arg;
    
    // printf("Thread id %d has index %d\n", tid, index);
	
    // Scan algorithm 
    // Part I: Up-sweep (reduce)
    int step = 0;
    int off = 1;
    for (int d = v.size()/2; d >0; d = d/2) {
        pthread_barrier_wait(&barrier);
        int iSum = (d/thread_cnt) > 0?(d/thread_cnt):1;
        for (int i = 0; i<iSum;i++) {
            // TODO!!
            if (index<d && ((i*thread_cnt)<d || i==0)) {
                int left  = (off*(2*index+1)-1)+(i*thread_cnt*off*2);//i*thread_cnt*2;
                int right = (off*(2*index+2)-1)+(i*thread_cnt*off*2);//i*thread_cnt*2;
                v[right] += v[left];
            } 
        }
        off *= 2;
        step +=1;
    }

    if (index == 0) {
        int last = v.size()-1;
        // Remember full sum
        full_sum = v[last];
        // Init identity element 
        v[last] = 0;
    } 

    // Part II: 
    for (int d = 1; d < v.size(); d = d*2) {
        off /= 2;
        pthread_barrier_wait(&barrier);
        int iSum = (d/thread_cnt) > 0?(d/thread_cnt):1;
        for (int i = 0; i<iSum;i++) {
            // TODO!!!
            if (index<d && ((i*thread_cnt)<d || i==0)) {
                int left  = (off*(2*index+1)-1)+(i*thread_cnt*off*2);//i*thread_cnt*2;;
                int right = (off*(2*index+2)-1)+(i*thread_cnt*off*2);//i*thread_cnt*2;;
                int t = v[left];
                v[left] = v[right];
                v[right] += t;
            }
        }
    }
    // pthread_barrier_wait(&barrier);
    return NULL;
}


int main( int argc, char * argv[] )
{
	// Parse arguments
	const uint element_cnt = argc >= 2 ? atoi(argv[1]) : 8;
	thread_cnt = argc >= 3 ? atoi(argv[2]) : THREAD_CNT;
    struct timespec t0, t1;
    
    if (ceil(log2(element_cnt)) != floor(log2(element_cnt))) exit(3);

    // Create threads and fill tid/index map
    pthread_t threads[thread_cnt];    
    pthread_barrier_init(&barrier, NULL, thread_cnt);
    if (DEFAULT_VALS) v = {1,2,3,4,5,6,7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    else for (int i = 0; i<element_cnt; i++) v.push_back(MIN + (rand() % static_cast<int>(MAX - MIN + 1)));
    if (DEBUGGING) printVector(v);

    vector<int> thread_cnts = {1,2,4,8,12,16,24,32,40,48};

    //for(int j=0; j<thread_cnts.size();j++){
        //thread_cnt = thread_cnts[j];
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for(uint i = 0; i < thread_cnt; ++i) 
        {
            args_t args = { NULL };        
            if(pthread_create(&threads[i], NULL, thread_fn_pthread, &args)) {
                exit(2);
            }
            ids.insert(pair<int, int>(threads[i], i));
        }
        if (DEBUGGING) printMap(ids);
        
        for(uint i = 0; i < thread_cnt; ++i) if(pthread_join(threads[i], NULL)) exit(3);
        
        v = preScan2Scan(v, full_sum);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (DEBUGGING) printVector(v);    
        double time = (double) (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e6;
        cout << endl << thread_cnt << "," << time << endl;
    //}
    return 0;
}