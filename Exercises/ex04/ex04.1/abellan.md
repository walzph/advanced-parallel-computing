In the paper "GLocks: Efﬁcient Support for Highly-Contended Locks in Many-Core CMPs" José L. Abellán and his colleagues describe GLocks, which is their approach to lock synchronization.

Because synchronization among many-core CMPs is a key limitor to performance and scalability often caused by shared variables foor access coordination, the authors main contribution is the concept of a hardware-supported lock mechanism. Because GLocks uses a dedicated network for synchronization, the technology skips the memory hierearchy. 

To prove the viability of their highly-contended lock concept, the authors provide a comprehensive comparison against common shared-memory locks. This evaluations shows that GLocks reduces power consumption and execution time.

We strongly agree the authors proposal. They showed a reasonable lock optimization approach which can be seen as a candidate to leverage any multi-core application.   