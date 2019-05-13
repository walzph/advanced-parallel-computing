In the paper "Cost-Effective Parallel Computing", David A. Wood and Mark D. Hill show in which cases multiprocessor systems are financially worthwhile, even if no linear speedup is available.

They show that this mainly depends on memory. If a processor does not provide enough performance, you should use multiple processors instead to utilize the full memory capacity and bandwidth.

I accept their statement, even though the paper was written a few years ago and parallel systems are now needed to keep track of Amdahl's Law, as the performance of a single processor can no longer be achieved by increasing the clock frequency. I think it's good that even at that time they didn't just use the pure speedup as a basis, but also referred to the total costs for the respective performance.