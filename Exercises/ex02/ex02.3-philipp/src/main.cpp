#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <pthread.h>


#include <chCommandLine.h>
#include <chTimer.hpp>

using namespace std;

//
// Function Prototypes
//
void *loadMem(void *arguments);
void printHelp(char * programName);

struct arg_struct {
	long long unsigned addr_start;
	int length;
	int thread_id;
};

int main( int argc, char * argv[] )
{

    bool optShowHelp = chCommandLineGetBool("h", argc, argv);
	if ( !optShowHelp )
		optShowHelp = chCommandLineGetBool("help", argc, argv);

	if ( optShowHelp ) {
		printHelp ( argv[0] );
		exit (0);
	}

    int ELEMENTS = 1024;
    chCommandLineGet(&ELEMENTS, "e", argc, argv);
    chCommandLineGet(&ELEMENTS, "elements", argc, argv);

	volatile char *buffer = (char *) malloc (ELEMENTS * sizeof(char));
  	volatile long long unsigned address = (long long unsigned) buffer;
  	if (buffer==NULL) exit (1);

  	for (int n=0; n<ELEMENTS; n++)
    	buffer[n]=n%26+'a';
	
	int MAX_THREADS=1024;
	chCommandLineGet(&MAX_THREADS, "t", argc, argv);
	chCommandLineGet(&MAX_THREADS, "max_threads", argc, argv);
	
	for (int k=MAX_THREADS; k>0; k=k/2) {
		int NUM_THREADS = k;
	
		volatile struct arg_struct args [NUM_THREADS];
		
		pthread_t threads[NUM_THREADS];
		int rc;
		int i;
		chTimerTimestamp start, stop;

		chTimerGetTime(&start);

		for( i = 0; i < NUM_THREADS; i++ ) {
			// cout << "main() : creating thread, " << i << endl;
			int length = (ELEMENTS * sizeof(char)) / NUM_THREADS;
			args[i].addr_start = address + i * length;
			args[i].length = length;
			args[i].thread_id = i;
			rc = pthread_create(&threads[i], NULL, loadMem, (void *)&args[i]);
			
			if (rc) {
				cout << "Error:unable to create thread," << rc << endl;
				exit(-1);
			}
		}

		chTimerGetTime(&stop);

		double microseconds = 1e6 * chTimerElapsedTime(&start, &stop);

		cout << k << "," << microseconds<<endl; 
	}
	pthread_exit(NULL);

	return 0;
}

void *loadMem(void *arguments) {
	volatile struct arg_struct *args = (struct arg_struct *) arguments;

	// cout << "Thread with id " << (int) args->thread_id 
	// 	 << " starts at " <<  args->addr_start 
	// 	 << " reads " << args->length << "bytes" << endl;

	volatile char* buffer = reinterpret_cast<char*>(args->addr_start);
	for (int i=0; i<args->length;i++) {
		char c = buffer[i];
	}
	// cout << "Reading first value in buffer: " << buffer[0] << endl;

}

void printHelp(char * programName)
{
	std::cout 	
		<< "Usage: " << std::endl
		<< "  " << programName  << std::endl
		<< "  -d <dimension_size>|--dimension <dimension_size>" << std::endl
		<< "    The dimension of both squared matrix to be multiplied" << std::endl
		<< "  --info" << std::endl
		<< "    Print header info to console" << std::endl        
		<< "  --result" << std::endl
		<< "    Write matrices A, B and C to stdout " << std::endl
		<< "" << std::endl;
}