# Advanced Parallel Computing
*28.05.2019*  
**Students:**  
Jona Neef  
Nikolas Krätzschmar  
Philipp Walz  

## Exercise 5

### 5.1 Reading

#### Shucai Xiao (Virginia Tech, US); Wu-chun Feng (Virginia Tech, US), Inter-Block GPU Communication via Fast Barrier Synchronization, IPDPS 2010.

In their paper Shucai Xiao and Wu-chun Feng describe an approach with which it is possible to perform inter-block communication on GPUs without barrier synchronization. This normally generates an enormous overhead, since it can only be done via the global memory, and thus the CPU.

The key insight was that using lock-based synchronizations (using mutex and atomics) and lock-free synchronizations (two arrays with synchronization variables) it is definitely possible to get better performance. Since the synchronization time for certain algorithms takes more than half of the total execution time, it is even more important to reduce the synchronization time instead of just optimizing the actual execution.

I accept the work of the authors, because I also experienced the problem of overhead via barrier synchronization in CUDA programs. In my opinion, the optimization of synchronization time plays a greater role in many programs than the optimization of the execution itself, which is why it is a great loss that only a fraction of the papers deal with it.


#### Torsten Hoefler, Torsten Mehlan, Frank Mietke and Wolfgang Rehm, Fast Barrier Synchronization for InfiniBand, CAC 2006, co-located with IPDPS 2006.

In the paper "Fast Barrier Synchronization for InfiniBand" Torsten Hoefler et. al. describe their approach to model modern offloading based interconnect networks such as IniniBand. Based on this newly introduced LoP model, they propose a new low-latency, implicit, hardware-parallel method for barrier synchronization. 

The authors main contirbution however lies in the introduction of a more accurate model for offloding interconnects used in HPC, which shows that implicit hardware-parallelism (i.e. sending 2 messages consecutively rather than with some time between) significantly increases barrier performance.

Because their research on barrier implementations showed that the dissemination algorithm is best used in cluster networks, they propose then-way Dissemination Principle which makes use of their research findings.

We strongly agree with the authors work. As scientific computing depends on larger data sets, a scalable execution on large clusters becomes indispensable. This also makes fast synchronization techniques of prime significance.
Interestingly because additional hardware offload interconnects are even more widely used nowadays, their work is equally relevant today. 

