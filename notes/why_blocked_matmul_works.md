# why blocked matmul works

> **memory hierarchy management**.

- cpus have caches like L1, L2 and L3 and blocked matrix multiplication is designed in such a way that to keep the data in cache for as long as possible rather than storing it in ram

---

### **The Mechanics of the "Cache Nightmare"**

if data is stored in row order for both matrix A and B then the fetching of row A is not problem rather its quite efficient but fetching the column values from B is big headache and given is the reason:

When the CPU needs a single piece of data (like $B_{0,j}$), it doesn't just grab that one number. It fetches a **cache line**—a small chunk of contiguous memory (usually 64 bytes).

1. **The Fetch:** If you are accessing Matrix B in a column-major way ($B_{0,j}$, then $B_{1,j}$, then $B_{2,j}$), the CPU pulls a cache line containing $B_{0,j}$ and its horizontal neighbors (like $B_{0,j+1}, B_{0,j+2}$, etc.).
2. **The Jump:** Because the matrix is stored in **row-major order** in memory, the next element you actually need ($B_{1,j}$) is located far away—exactly one full row width away. It is almost certainly _not_ in the cache line you just fetched.
3. **The Eviction:** As the algorithm moves down the column, it keeps demanding new cache lines from RAM. Since the cache has limited space, it must start throwing out the oldest lines to make room for the new ones.
4. **The Result:** By the time the inner loop finishes the first column and moves to the next ($j+1$), the cache lines that were fetched at the very beginning (which actually contained the data for the start of the next column!) have been **evicted**.

---

**Spatial locality**: If a particular storage location is referenced at a particular time, then it is likely that nearby memory locations will be referenced in the near future. In this case it is common to attempt to guess the size and shape of the area around the current reference for which it is worthwhile to prepare faster access for subsequent reference.

basically being able to fetch elements from same cache line meant they are in our spatial locality.

---

### The solution:

**Blocked (or Tiled) multiplication** breaks the large matrices into smaller sub-matrices (blocks) that are small enough to fit entirely within the cache.

**How it optimizes performance:**

- **Temporal Locality:** Once a block of $A$ and a block of $B$ are loaded into the cache, the CPU performs all possible operations using that data before moving on. Instead of loading an element, using it once, and throwing it away, the CPU reuses it multiple times.
- **Reduced Traffic:** By working on blocks of size $B \times B$, you reduce the total number of times data must be read from main memory. In a naive approach, an element might be loaded $N$ times; in a blocked approach, it is loaded roughly $N/B$ times.

---

### trying out *"transposing B"* solution but why blocked matmul will strill outperform it?

Transposing matrix B before multiplying—so that you compute the dot product of a row of A and a row of $B^T$—is actually a very common and smart optimization.

By doing this, you perfectly solve the **spatial locality** problem. The CPU fetches a cache line for $B^T$, and your algorithm uses every single number in that cache line because it's reading horizontally. There are no wasted fetches.

However, **yes, blocked matrix multiplication will still heavily outperform this transposed naive approach on large matrices.** Here is exactly how and why that happens. It comes down to the difference between fixing _spatial_ locality (fetching efficient chunks) and fixing _temporal_ locality (reusing what you fetched).

### The Problem with the Transposed Approach: Zero Data Reuse.

Let's look at what happens when you multiply $A \times B^T$ (where both are $N \times N$ matrices, say $4000 \times 4000$).

1. You take the **first row of A**. You hold it in your L1 cache.
2. You multiply it by the **first row of $B^T$**. You get $C_{0,0}$.
3. You multiply it by the **second row of $B^T$**. You get $C_{0,1}$.
4. ...You do this for all $N$ rows of $B^T$.

**Here is the catch:** By the time you finish calculating just the first row of matrix C, you have loaded the _entirety_ of matrix $B^T$ from main memory (RAM) into the cache, and then pushed it right back out because it's too big to fit.

Now, you move to the **second row of A**. To calculate its dot products, you have to load the _entirety_ of matrix $B^T$ from RAM all over again!

In the transposed naive approach, you must fetch the entire matrix $B^T$ from slow main memory $N$ separate times. For a $4000 \times 4000$ matrix, that means moving hundreds of gigabytes of data back and forth from RAM, even though the matrix itself is only a few megabytes. You are starved by **Memory Bandwidth**.

### How Blocking Fixes This (Arithmetic Intensity):

Blocked matrix multiplication looks at this and says: _"If I'm going to spend the time bringing a piece of Matrix A and a piece of Matrix B into the fast L1 cache, I am not going to let them leave until I have squeezed every possible math operation out of them."_

This introduces a concept called **Arithmetic Intensity**: the ratio of math operations performed to bytes of memory fetched.

#### The Transposed Naive Math:

For every 2 numbers you fetch from RAM (one from A, one from $B^T$), you do exactly 2 math operations (one multiply, one add).

- **Ratio:** $\approx 1$ operation per memory fetch. The CPU spends most of its time waiting for RAM.
    

#### The Blocked Math:

Instead of full rows, imagine we load a small $b \times b$ block of A and a $b \times b$ block of B into the cache. (Let's say $b = 64$).

- We load $64 \times 64 = 4096$ numbers from A.
- We load $64 \times 64 = 4096$ numbers from B.
- Total memory fetches = $8192$.
- But how many math operations can we do with those two blocks before we need new data? Matrix multiplication of a $b \times b$ block requires $2b^3$ operations. That's $2 \times 64^3 \approx 524,000$ operations!
- **Ratio:** $\approx 64$ operations per memory fetch.

