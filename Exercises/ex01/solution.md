# Advanced Parallel Computing
*29.04.2019*  

**Students:**  
Jona Neef  
Nikolas Krätzschmar  
Philipp Walz  

## Exercise 1: Reading

### Shekhar Borkar and Andrew A. Chien. 2011. The future of microprocessors. Commun. ACM 54, 5 (May 2011), 67-77.

In their paper "The future of microprocessors" Borkar and Chien reflect and project the macro trends that will shape the future of microprocessors. 

The paper starts by illustrating the most important microprocessor developments over the past 20 years, such as the classical transistor-speed scaling which delivered exponential performance increase: Every two years transistor integration doubled, circuit speed increased by 40%, while the system power consumption was kept the same. Furthermore, the authors show performance effects and limitations of core microarchitecture techniques (pipelining, branch prediction, out-of-order execution, speculation) as well as of cache memory architectures.

However, the authors show that these methods won't be viable in the next 20 years anymore due to a limited total energy budget. According to the authors, the ultimate denominator in this new era of microprocessors will be energy efficiency. Henceforth various challenges and near- and long-term solutions are proposed. These challenges are grouped in logic organization-, data movement-, circuit- and software challenges. To solve these challenges under limited energy usage, the authors propose methods such as large-scale parallelism with heterogeneous cores, customized accelerators or efficient data orchestration by using memory hierarchies and new types of interconnect 

Finally, the authors give a prospect on new computing regimes apart from Si CMOS, such as computing with carbon nanotubes or quantum electronics. Borkar and Chien conclude with the notion that although these challenges are complex, they are comparatively small in contrast to what lies ahead with alternative computing technologies. 


### Peter J. Denning and Ted G. Lewis. 2016. Exponential laws of computing growth. Commun. ACM 60, 1 (December 2016), 54-65.

In their paper "Exponential Laws of Computing Growth", Peter J. Denning and Ted G. Lewis describe an approach that goes beyond simply increasing computing power by increasing clock speed or using multi-core processors. Instead, they argue that such exponential growth would not have been possible if all three levels of the computing ecosystem (chip, system, adopting community) had not grown together.

Furthermore, it is also important that technology jumps are made by companies when old technologies no longer yield a return. Exponential growth can only work with new technologies.

In our opinion, it is remarkable how exponential growth of computing power continues to grow as new ideas and technologies are developed. While the shift from single-core to multi-core processors was a necessary step to sustain exponential growth, it also leads to much greater complexity in software development.