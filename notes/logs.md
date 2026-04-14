# logs

### date: 15/04/2026

- using sum variable to store the matmul cause loss of performane in pointer implementation. rather if we directly add values in C matrix then it gives massive performance gain like from 1560ms to 190ms.
- but the same thing backfires on vector implementation and loses perfomance.
- gigaflops formula: 

$$
\text{GigaFLOPS} = \frac{2 \times M \times N \times K}{\text{Time in Seconds} \times 10^9}
$$

### date: 14/04/2026

- was using -O2 flag for adv_boxed_parallel_matmul.cpp while using -O0 for all other implementations. this was probably a mistake cause then compiler added some more optimization which i don't know nothing about and it also was unfair comparison between implementation that is not optimized by compiler vs implementation that is optimized by compiler.
- boxed matmul is good it only gives like 400ms of performance gain for 1024 dimension matrix while comparing it with naive matmul with pointer version to store matrix.
- the real gain happens when we use openmp to parallelize our matmul i.e each block is handled by different thread. this gives us like 3400ms of performance gain when comparing with naive matmul with pointer version to store matrix.
