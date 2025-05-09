
# Course Notes: Design and Analysis of Algorithms

## Unit 1: Foundation of Algorithm Analysis

### 1.1 Algorithm and its properties, RAM model, Time and Space Complexity, detailed analysis of algorithms (Like factorial algorithm), Concept of Aggregate Analysis

#### Algorithm
*   **Definition:** A well-defined computational procedure that takes some value, or set of values, as input and produces some value, or set of values, as output.
*   **Properties:**
    1.  **Input:** Zero or more quantities are externally supplied.
    2.  **Output:** At least one quantity is produced.
    3.  **Definiteness:** Each instruction is clear and unambiguous.
    4.  **Finiteness:** Terminates after a finite number of steps.
    5.  **Effectiveness:** Every instruction must be basic enough to be carried out.

#### RAM (Random Access Machine) Model
*   A theoretical model of computation.
*   **Assumptions:**
    1.  Simple instructions take one time step.
    2.  Loops and subroutines are not simple instructions.
    3.  Unlimited memory, one time step access.
    4.  Sequential execution.
*   Simplifies analysis by ignoring hardware complexities.

#### Time and Space Complexity
*   **Time Complexity:** Amount of time an algorithm takes in terms of input size (n). Counts elementary operations.
*   **Space Complexity:** Amount of memory an algorithm requires in terms of input size (n).
    *   **Instruction Space:** For compiled instructions.
    *   **Data Space:** For constants, variables (input space + auxiliary space).
    *   Often interested in *auxiliary space complexity*.

#### Detailed Analysis of Algorithms (Example: Factorial Algorithm)

*   **Iterative Factorial:**
    ```
    function factorial_iterative(n):
        if n < 0:
            return error "Input must be non-negative"
        if n == 0:
            return 1
        result = 1
        for i from 1 to n:
            result = result * i
        return result
    ```
    *   **Time Analysis:** T(n) ≈ c1 + c2*n. O(n).
    *   **Space Analysis:** S(n) = O(1) (constant auxiliary space).

*   **Recursive Factorial:**
    ```
    function factorial_recursive(n):
        if n < 0:
            return error "Input must be non-negative"
        if n == 0:
            return 1
        else:
            return n * factorial_recursive(n-1)
    ```
    *   **Time Analysis:** T(n) = T(n-1) + c. O(n).
    *   **Space Analysis:** S(n) = O(n) (recursion stack depth).

#### Concept of Aggregate Analysis
*   Finds the average cost of an operation over a sequence.
*   A single operation might be expensive, but average cost is low.
*   Example: Dynamic array `push_back`. Most are O(1), occasional reallocation O(k). Amortized O(1).
*   Amortized cost = Total cost of N operations / N.

### 1.2 Asymptotic Notations: Big-O, Big-Ω and Big-Ө Notations their Geometrical Interpretation and Examples.

#### Big-O Notation (O): Asymptotic Upper Bound
*   **Definition:** `f(n) = O(g(n))` if ∃ positive constants `c`, `n₀` such that `0 ≤ f(n) ≤ c * g(n)` for all `n ≥ n₀`.
*   **Meaning:** `f(n)` grows no faster than `g(n)`.
*   **Example:** `3n² + 2n + 5 = O(n²)`.
*   Often for worst-case analysis.

#### Big-Omega Notation (Ω): Asymptotic Lower Bound
*   **Definition:** `f(n) = Ω(g(n))` if ∃ positive constants `c`, `n₀` such that `0 ≤ c * g(n) ≤ f(n)` for all `n ≥ n₀`.
*   **Meaning:** `f(n)` grows at least as fast as `g(n)`.
*   **Example:** `3n² + 2n + 5 = Ω(n²)`.
*   Often for best-case analysis or problem lower bounds.

#### Big-Theta Notation (Ө): Asymptotic Tight Bound
*   **Definition:** `f(n) = Ө(g(n))` if ∃ positive constants `c₁`, `c₂`, `n₀` such that `0 ≤ c₁ * g(n) ≤ f(n) ≤ c₂ * g(n)` for all `n ≥ n₀`.
*   **Meaning:** `f(n)` grows at the same rate as `g(n)`. `f(n) = O(g(n))` AND `f(n) = Ω(g(n))`.
*   **Example:** `3n² + 2n + 5 = Ө(n²)`.
*   Most precise asymptotic description.

#### Geometrical Interpretation
*   Plot `f(n)` and `g(n)`.
*   **O:** `f(n)` on or below `c * g(n)` for `n ≥ n₀`.
*   **Ω:** `f(n)` on or above `c * g(n)` for `n ≥ n₀`.
*   **Ө:** `f(n)` "sandwiched" between `c₁ * g(n)` and `c₂ * g(n)` for `n ≥ n₀`.
    ```
    ^ Growth Rate
    |
    |        c₂*g(n)
    |       /
    |      / f(n)
    |     /
    |  c₁*g(n)
    |  /
    | /
    |/__________________> n (Input size)
         n₀
    ```

#### Common Growth Rates (slowest to fastest):
*   `O(1)`: Constant
*   `O(log n)`: Logarithmic
*   `O(√n)`: Square root
*   `O(n)`: Linear
*   `O(n log n)`: Log-linear
*   `O(n²)`: Quadratic
*   `O(n³)`: Cubic
*   `O(2ⁿ)`: Exponential
*   `O(n!)`: Factorial

### 1.3 Recurrences: Recursive Algorithms and Recurrence Relations, Solving Recurrences (Recursion Tree Method, Substitution Method, Application of Masters Theorem)

*   **Recursive Algorithms:** Algorithms calling themselves.
*   **Recurrence Relation:** Equation describing a function via its value on smaller inputs.
    *   Example (Merge Sort): `T(n) = 2T(n/2) + O(n)`
    *   Needs base cases, e.g., `T(1) = O(1)`.

#### Solving Recurrences:

1.  **Substitution Method:**
    *   Guess solution form.
    *   Use induction to verify and find constants.
    *   Example: `T(n) = 2T(n/2) + n`. Guess `T(n) = O(n log n)`. Show `T(n) ≤ c n log n`.
        Assume `T(n/2) ≤ c (n/2) log(n/2)`.
        `T(n) ≤ 2(c (n/2) log(n/2)) + n = c n log n - c n + n`.
        Need `c n log n - c n + n ≤ c n log n`, implies `n ≤ c n`, so `c ≥ 1`.

2.  **Recursion Tree Method:**
    *   Tree representing cost of subproblems. Sum costs per level, then all levels.
    *   Example: `T(n) = 2T(n/2) + n`
        ```
                       n                  Cost: n
                     /   \
                 T(n/2)   T(n/2)            Cost: n/2 + n/2 = n
                  / \     / \
             T(n/4)...T(n/4)              Cost: n/4 * 4 = n
               ...
             T(1) ...                     Cost: n * T(1)
        ```
        *   Depth: `log₂n`. Cost at each level `n`. Total: `O(n log n)`.

3.  **Master's Theorem:**
    *   For recurrences: `T(n) = a T(n/b) + f(n)` where `a ≥ 1, b > 1`.
    *   Compares `f(n)` with `n^(log_b a)`.
        *   **Case 1:** If `f(n) = O(n^(log_b a - ε))` for `ε > 0`, then `T(n) = Ө(n^(log_b a))`.
        *   **Case 2:** If `f(n) = Ө(n^(log_b a))`, then `T(n) = Ө(n^(log_b a) * log n)`.
        *   **Case 3:** If `f(n) = Ω(n^(log_b a + ε))` for `ε > 0`, AND `a * f(n/b) ≤ c * f(n)` for `c < 1` (regularity condition), then `T(n) = Ө(f(n))`.
    *   Example: `T(n) = 2T(n/2) + n`. `a=2, b=2, f(n)=n`. `n^(log₂ 2) = n`. Case 2. `T(n) = Ө(n log n)`.

---

## Unit 2: Iterative Algorithms

### 2.1 Basic Algorithms: Algorithm for GCD, Fibonacci Number and analysis of their time and space complexity

#### Greatest Common Divisor (GCD): Euclidean Algorithm
*   **Problem:** Largest integer dividing `a` and `b`.
*   **Algorithm (Euclid's):**
    ```
    function GCD(a, b):
        while b != 0:
            temp = b
            b = a mod b
            a = temp
        return a
    ```
*   **Analysis:**
    *   Time Complexity: `O(log(min(a,b)))`.
    *   Space Complexity: `O(1)` (iterative).

#### Fibonacci Numbers
*   Sequence: `F(0)=0, F(1)=1, F(n) = F(n-1) + F(n-2)`.

*   **Recursive Algorithm (naive):**
    ```
    function fib_recursive(n):
        if n <= 1: return n
        else: return fib_recursive(n-1) + fib_recursive(n-2)
    ```
    *   Time Complexity: `O(2ⁿ)`.
    *   Space Complexity: `O(n)` (stack).

*   **Iterative Algorithm (Dynamic Programming):**
    ```
    function fib_iterative(n):
        if n <= 1: return n
        a = 0; b = 1
        for i from 2 to n:
            c = a + b; a = b; b = c
        return b
    ```
    *   Time Complexity: `O(n)`.
    *   Space Complexity: `O(1)`.

*   **Matrix Exponentiation Method (Advanced):**
    *   Time Complexity: `O(log n)`.
    *   Space Complexity: `O(1)` or `O(log n)`.

### 2.2 Searching Algorithms: Sequential Search and its analysis

#### Sequential Search (Linear Search)
*   **Problem:** Find target `x` in array `A`.
*   **Algorithm:**
    ```
    function sequential_search(A, n, x):
        for i from 0 to n-1:
            if A[i] == x: return i
        return -1
    ```
*   **Analysis (unsorted array):**
    *   Time Complexity:
        *   Best Case: `O(1)`.
        *   Worst Case: `O(n)`.
        *   Average Case: `O(n)`.
    *   Space Complexity: `O(1)`.

### 2.3 Sorting Algorithms: Bubble, Selection, and Insertion Sort and their Analysis

#### Bubble Sort
*   **Idea:** Repeatedly compare adjacent elements, swap if wrong order.
*   **Algorithm:**
    ```
    function bubble_sort(A, n):
        swapped = true
        for i from 0 to n-2:
            swapped = false
            for j from 0 to n-2-i:
                if A[j] > A[j+1]:
                    swap(A[j], A[j+1])
                    swapped = true
            if not swapped: break
    ```
*   **Analysis:**
    *   Time: Best `O(n)`, Worst/Average `O(n²)`.
    *   Space: `O(1)`.
    *   Stable: Yes.

#### Selection Sort
*   **Idea:** Find min from unsorted part, place at beginning.
*   **Algorithm:**
    ```
    function selection_sort(A, n):
        for i from 0 to n-2:
            min_index = i
            for j from i+1 to n-1:
                if A[j] < A[min_index]: min_index = j
            if min_index != i: swap(A[i], A[min_index])
    ```
*   **Analysis:**
    *   Time: Best/Worst/Average `O(n²)`.
    *   Space: `O(1)`.
    *   Stable: No (by default).

#### Insertion Sort
*   **Idea:** Build sorted array one item at a time. Insert item into correct sorted position.
*   **Algorithm:**
    ```
    function insertion_sort(A, n):
        for i from 1 to n-1:
            key = A[i]
            j = i - 1
            while j >= 0 and A[j] > key:
                A[j+1] = A[j]
                j = j - 1
            A[j+1] = key
    ```
*   **Analysis:**
    *   Time: Best `O(n)`, Worst/Average `O(n²)`.
    *   Space: `O(1)`.
    *   Stable: Yes. Adaptive.

---

## Unit 3: Divide and Conquer Algorithms

Strategy: Divide, Conquer, Combine.

### 3.1 Searching Algorithms: Binary Search, Min-Max Finding and their Analysis

#### Binary Search
*   **Prerequisite:** Sorted array.
*   **Idea:** Compare with middle. Recurse on left/right half.
*   **Algorithm (Iterative):**
    ```
    function binary_search_iterative(A, n, x):
        low = 0; high = n - 1
        while low <= high:
            mid = low + (high - low) // 2
            if A[mid] == x: return mid
            else if A[mid] < x: low = mid + 1
            else: high = mid - 1
        return -1
    ```
*   **Analysis:**
    *   Time: `T(n) = T(n/2) + O(1)`. `Ө(log n)`. Best O(1).
    *   Space: Iterative `O(1)`, Recursive `O(log n)`.

#### Min-Max Finding
*   **Problem:** Find min and max elements.
*   **Naive:** `2(n-1)` comparisons. `O(n)`.
*   **Divide and Conquer:** `T(n) = 2T(n/2) + 2`. Approx `3n/2 - 2` comparisons.
*   **Optimized Pairwise:** Process pairs. `3n/2` comparisons. `O(n)`.
*   **Analysis for D&C:** Time `O(n)`, Space `O(log n)`.

### 3.2 Sorting Algorithms: Merge Sort, Quick Sort (Best, Worst, Average Case), Heap Sort (Heapify, Build Heap, Heap Sort Algorithms), Randomized Quick sort.

#### Merge Sort
*   **Idea:** Divide array, recursively sort halves, merge sorted halves.
*   **Merge Operation (Conceptual):**
    ```
    function merge(A, L, M, R, arr):
        // Create LeftSub, RightSub from arr[L..M] and arr[M+1..R]
        // Merge LeftSub, RightSub back into arr[L..R] comparing elements
    ```
    Takes `O(n)` time for `n` elements.
*   **Algorithm:**
    ```
    function merge_sort(A, L, R):
        if L < R:
            M = L + (R - L) // 2
            merge_sort(A, L, M)
            merge_sort(A, M + 1, R)
            merge(A, L, M, R)
    ```
*   **Analysis:**
    *   Time: `T(n) = 2T(n/2) + O(n)`. `Ө(n log n)` (Best/Worst/Average).
    *   Space: `O(n)` (auxiliary array).
    *   Stable: Yes.

#### Quick Sort
*   **Idea:** Pick pivot, partition array (smaller elements left, larger right), recurse.
*   **Partition (Lomuto scheme, pivot = A[high]):**
    ```
    function partition(A, low, high):
        pivot = A[high]; i = low - 1
        for j from low to high - 1:
            if A[j] <= pivot: i++; swap(A[i], A[j])
        swap(A[i+1], A[high])
        return i + 1
    ```
    Takes `O(n)` time.
*   **Algorithm:**
    ```
    function quick_sort(A, low, high):
        if low < high:
            pi = partition(A, low, high)
            quick_sort(A, low, pi - 1)
            quick_sort(A, pi + 1, high)
    ```
*   **Analysis:**
    *   Time:
        *   Worst: `O(n²)`. Recurrence `T(n) = T(n-1) + O(n)`.
        *   Best: `O(n log n)`. Recurrence `T(n) = 2T(n/2) + O(n)`.
        *   Average: `O(n log n)`.
    *   Space: Average `O(log n)`, Worst `O(n)` (recursion stack).
    *   Stable: No. In-place.

#### Randomized Quick Sort
*   **Idea:** Choose pivot randomly or shuffle array.
*   **Analysis:** Expected Time `O(n log n)`. Worst case `O(n^2)` highly unlikely.

#### Heap Sort
*   Uses Binary Heap (complete binary tree, heap property: Max-Heap or Min-Heap).
*   Array representation: Parent `(i-1)/2`, Left `2i+1`, Right `2i+2`.

*   **Heapify (Max-Heapify):** Restore heap property at node `i` if children are heaps.
    ```
    function max_heapify(A, n, i):
        largest = i; left = 2*i + 1; right = 2*i + 2
        if left < n and A[left] > A[largest]: largest = left
        if right < n and A[right] > A[largest]: largest = right
        if largest != i:
            swap(A[i], A[largest])
            max_heapify(A, n, largest)
    ```
    Time: `O(log n)`.

*   **Build Heap (Build Max-Heap):** Convert array to max-heap.
    ```
    function build_max_heap(A, n):
        start_index = (n // 2) - 1
        for i from start_index down to 0: max_heapify(A, n, i)
    ```
    Time: `O(n)`.

*   **Heap Sort Algorithm:**
    1.  Build Max-Heap from array (`O(n)`).
    2.  Repeat `n-1` times:
        a. Swap root (max element `A[0]`) with last element `A[heap_size-1]`.
        b. Reduce `heap_size`.
        c. `max_heapify` root `A[0]` of reduced heap (`O(log n)`).
*   **Analysis:**
    *   Time: `O(n) + O(n log n) = O(n log n)` (Best/Worst/Average).
    *   Space: `O(1)` (in-place, excluding `O(log n)` for recursive heapify stack).
    *   Stable: No.

### 3.3 Order Statistics: Selection in Expected Linear Time, Selection in Worst Case Linear Time and their Analysis.

#### Order Statistic Problem (Selection Problem)
*   Find `i`-th smallest element in unsorted array.

#### Selection in Expected Linear Time (Randomized Select)
*   **Idea:** Use `randomized_partition` (like Quick Sort). Recurse on one side.
*   **Algorithm (`randomized_select`):**
    ```
    function randomized_select(A, p, r, i): // Find i-th smallest in A[p..r]
        if p == r: return A[p]
        q = randomized_partition(A, p, r) // Pivot index
        k = q - p + 1 // Rank of pivot
        if i == k: return A[q]
        else if i < k: return randomized_select(A, p, q - 1, i)
        else: return randomized_select(A, q + 1, r, i - k)
    ```
*   **Analysis:** Worst Time `O(n²)`, Expected Time `O(n)`.

#### Selection in Worst-Case Linear Time (Median-of-Medians Algorithm)
*   Guarantees `O(n)` worst-case. More complex.
*   **Idea:** Deterministically choose a good pivot.
    1. Divide into groups of 5.
    2. Find median of each group.
    3. Recursively find median `x` of these `n/5` medians. Use `x` as pivot.
    4. Partition around `x`. Recurse on appropriate side.
*   Pivot `x` guarantees subproblem size ≤ `7n/10`.
*   Recurrence: `T(n) ≤ T(n/5) + T(7n/10) + O(n)`. Solves to `O(n)`.
*   Higher constant factors than Randomized Select.

---

## Unit 4: Greedy Algorithms

### 4.1 Optimization Problems and Optimal Solution, Introduction of Greedy Algorithms, Elements of Greedy Strategy.

*   **Optimization Problems:** Find best solution (min/max value).
*   **Greedy Algorithms:** Make locally optimal choice hoping for global optimum.
*   **Elements of Greedy Strategy:**
    1.  **Greedy Choice Property:** Locally optimal choice leads to global optimum.
    2.  **Optimal Substructure:** Optimal solution contains optimal solutions to subproblems.

### 4.2 Greedy Algorithms: Fractional Knapsack, Job sequencing with Deadlines, Kruskal’s Algorithm, Prims Algorithm, Dijkstra’s Algorithm and their Analysis

#### Fractional Knapsack Problem
*   **Problem:** `n` items (weight `w_i`, value `v_i`), knapsack capacity `W`. Take fractions. Maximize value.
*   **Greedy:** Sort by `v_i/w_i` (value per unit weight) descending. Take as much as possible of highest ratio items.
*   **Analysis:** `O(n log n)` (for sorting).

#### Job Sequencing with Deadlines
*   **Problem:** `n` jobs (profit `p_i`, deadline `d_i`), unit time per job. Maximize profit, meet deadlines.
*   **Greedy:** Sort jobs by profit descending. Schedule job in latest possible free slot `≤ d_j`.
*   **Analysis:** `O(n log n + n * D_max)`. Can be `O(n log n)` with DSU for slot finding.

#### Minimum Spanning Tree (MST) Algorithms
*   Problem: Find tree connecting all vertices in weighted graph with min total edge weight.

*   **Kruskal’s Algorithm:**
    *   **Greedy:** Add next lightest edge if it doesn't form a cycle.
    *   **Algorithm:** Sort edges. Use DSU to track connected components.
    *   **Analysis:** `O(E log E)` or `O(E log V)`. Space `O(V+E)`.

*   **Prim’s Algorithm:**
    *   **Greedy:** Grow MST from start vertex. Add lightest edge connecting vertex in MST to vertex outside.
    *   **Algorithm:** Use Min-Priority Queue (PQ) for vertices outside MST, prioritized by min edge weight to MST.
    *   **Analysis (Binary Heap PQ):** `O(E log V)`.
    *   **Analysis (Fibonacci Heap PQ):** `O(V log V + E)`. Space `O(V+E)`.

#### Dijkstra's Algorithm (Single-Source Shortest Paths)
*   **Problem:** Weighted graph (non-negative weights), source `s`. Find shortest paths from `s` to all others.
*   **Greedy:** Similar to Prim's. Maintain distances `d[v]`. Select unvisited `u` with min `d[u]`, update neighbors.
*   **Algorithm:** Use Min-PQ for unvisited vertices, prioritized by `d[v]`.
*   **Analysis:** Same as Prim's: `O(E log V)` (Binary Heap), `O(V log V + E)` (Fibonacci Heap).
*   **Note:** Fails with negative edge weights.

### 4.3 Huffman Coding: Purpose of Huffman Coding, Prefix Codes, Huffman Coding Algorithm and its Analysis

*   **Purpose:** Lossless data compression. Variable-length codes (shorter for frequent chars).
*   **Prefix Codes:** No codeword is a prefix of another. Allows unambiguous decoding. Represented by full binary tree.
*   **Huffman Algorithm:**
    *   **Greedy:** Repeatedly merge two least frequent characters/subtrees.
    *   **Algorithm:** Use Min-PQ for nodes (chars/subtrees) by frequency. Extract two min, combine into new node, insert back.
*   **Analysis:** `O(C log C)` for `C` characters. Generates optimal prefix code.

---

## Unit 5: Dynamic Programming

### 5.1 Greedy Algorithms vs Dynamic Programming, Recursion vs Dynamic Programming, Elements of DP Strategy

*   **Greedy vs. DP:**
    *   Greedy: Local choice before subproblems. Not always optimal.
    *   DP: Uses solutions to all subproblems. Usually optimal.

*   **Recursion vs. DP:**
    *   Recursion: Breaks problem down. Can recompute subproblems.
    *   DP: Addresses overlapping subproblems by storing solutions (memoization/tabulation).

*   **Elements of DP Strategy:**
    1.  **Optimal Substructure:** Optimal solution to problem contains optimal solutions to subproblems.
    2.  **Overlapping Subproblems:** Recursive solution solves same subproblems multiple times.

*   **Steps to Develop DP:**
    1. Characterize optimal solution structure.
    2. Recursively define optimal solution value (recurrence).
    3. Compute bottom-up or top-down with memoization.
    4. (Optional) Construct solution.

### 5.2 DP Algorithms: Matrix Chain Multiplication, String Editing, Zero-One Knapsack Problem, Floyd Warshall Algorithm, Travelling Salesman Problem and their Analysis.

#### Matrix Chain Multiplication (MCM)
*   **Problem:** Sequence of `n` matrices. Find optimal parenthesization for min scalar multiplications.
*   **DP:** `m[i, j]` = min cost for `A_i...A_j`.
*   **Recurrence:** `m[i, j] = min_{i ≤ k < j} (m[i, k] + m[k+1, j] + p_{i-1}*p_k*p_j)`.
*   **Analysis:** Time `O(n³)`. Space `O(n²)`.

#### String Editing

*   **Longest Common Subsequence (LCS):**
    *   **Problem:** Given `X`, `Y`, find max length common subsequence.
    *   **DP:** `c[i, j]` = LCS length of `X_i`, `Y_j`.
    *   **Recurrence:**
        If `x_i == y_j`: `c[i,j] = c[i-1,j-1] + 1`.
        Else: `c[i,j] = max(c[i-1,j], c[i,j-1])`.
    *   **Analysis:** Time `O(m*n)`. Space `O(m*n)` (can be `O(min(m,n))`).

*   **Edit Distance (Levenshtein):**
    *   **Problem:** Min edits (insert, delete, substitute) to change `s1` to `s2`.
    *   **DP:** `dp[i][j]` = edit distance `s1[0..i-1]` and `s2[0..j-1]`.
    *   **Recurrence:**
        If `s1[i-1] == s2[j-1]`: `dp[i][j] = dp[i-1][j-1]`.
        Else: `dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])`.
    *   **Analysis:** Time `O(m*n)`. Space `O(m*n)`.

#### Zero-One Knapsack Problem
*   **Problem:** `n` items (weight `w_i`, value `v_i`), capacity `W`. Take whole item or not. Max value.
*   **DP:** `dp[i][j]` = max value using items `1..i` with capacity `j`.
*   **Recurrence:** `dp[i][j] = max(dp[i-1][j], v_i + dp[i-1][j - w_i])` if `j ≥ w_i`. Else `dp[i-1][j]`.
*   **Analysis:** Time `O(n*W)`. Space `O(n*W)` (can be `O(W)`). Pseudo-polynomial.

#### Floyd-Warshall Algorithm (All-Pairs Shortest Paths)
*   **Problem:** Weighted graph (can have neg edges, no neg cycles). Shortest path between all pairs.
*   **DP:** `dist[i][j][k]` = shortest path `i` to `j` using intermediates `{1..k}`.
    Optimized to `dist[i][j]` using iteration for `k`.
*   **Recurrence (optimized):**
    ```
    for k from 1 to V:
        for i from 1 to V:
            for j from 1 to V:
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    ```
*   **Analysis:** Time `O(V³)`. Space `O(V²)`.

#### Travelling Salesman Problem (TSP)
*   **Problem:** Visit `n` cities once, return to start, min total distance. NP-Hard.
*   **DP (Held-Karp):** `dp[S][j]` = shortest path from city 1, visiting all cities in set `S`, ending at `j ∈ S`.
*   **Recurrence:** `dp[S][j] = min_{i ∈ S, i ≠ j} (dp[S - {j}][i] + distance[i][j])`.
*   **Analysis:** Time `O(n² * 2ⁿ)`. Space `O(n * 2ⁿ)`. Exponential.

### 5.3 Memoization Strategy, Dynamic Programming vs Memoization

*   **Memoization (Top-Down DP):** Recursive calls, store/lookup subproblem solutions.
    ```
    memo = array init to -1
    function fib_memo(n):
        if n <= 1: return n
        if memo[n] != -1: return memo[n]
        memo[n] = fib_memo(n-1) + fib_memo(n-2)
        return memo[n]
    ```
*   **Tabulation (Bottom-Up DP) vs. Memoization:**
    | Feature             | Tabulation (Bottom-Up)                        | Memoization (Top-Down)                      |
    |---------------------|-----------------------------------------------|---------------------------------------------|
    | **Approach**        | Iterative, pre-defined order                  | Recursive, as needed                        |
    | **Subproblems**     | Solves all, maybe unneeded ones             | Solves only needed ones                     |
    | **Overhead**        | Loop                                          | Function call                               |
    | **Stack Overflow**  | No risk                                       | Risk for deep recursion                     |
    * Both usually have same asymptotic complexity.

---

## Unit 6: Backtracking

### 6.1 Concept of Backtracking, Recursion vs Backtracking

*   **Concept:** Algorithmic technique for constraint satisfaction problems. Builds candidates, abandons if invalid ("backtracks"). DFS on state-space tree.
    ```
    function backtrack(candidate):
        if is_solution(candidate): process_solution; return
        for each choice 'c' to extend candidate:
            if is_valid(candidate, c):
                add 'c' to candidate
                backtrack(candidate)
                remove 'c' from candidate // Backtrack!
    ```
*   **Recursion vs. Backtracking:**
    *   Backtracking is a specialized form of recursion for exploring possibilities, with constraints and undoing choices.
    *   All backtracking is recursive; not all recursion is backtracking.

### 6.2 Backtracking Algorithms: Subset-sum Problem, Zero-one Knapsack Problem, N-queen Problem and their Analysis.

#### Subset-Sum Problem
*   **Problem:** Given set `S`, target `T`. Is there a subset summing to `T`?
*   **Backtrack:** Include or exclude current element.
    ```
    function subset_sum_backtrack(S, target, index, current_sum, current_subset):
        if current_sum == target: print subset; return true
        if index >= S.length or current_sum > target: return false
        // Include S[index]
        current_subset.add(S[index])
        if subset_sum_backtrack(S, target, index + 1, current_sum + S[index], current_subset): return true
        current_subset.remove(S[index]) // Backtrack
        // Exclude S[index]
        if subset_sum_backtrack(S, target, index + 1, current_sum, current_subset): return true
        return false
    ```
*   **Analysis:** Time `O(2ⁿ)`. Space `O(n)`.

#### Zero-One Knapsack Problem (using Backtracking)
*   **Backtrack:** Include or exclude item. Prune if `current_weight > W`. Can use bounding function.
*   **Analysis:** Time `O(2ⁿ)`. Space `O(n)`.

#### N-Queen Problem
*   **Problem:** Place `N` queens on `N x N` board, no two attack.
*   **Backtrack:** Place queens row by row. Try all columns in current row. If safe, recurse. If leads to dead end, backtrack.
    ```
    function solve_NQueens(board, N, row):
        if row == N: print_solution; return true
        for col from 0 to N-1:
            if is_safe(board, N, row, col):
                board[row] = col // Place
                if solve_NQueens(board, N, row + 1): return true // if one solution
                board[row] = -1 // Backtrack
        return false

    function is_safe(board, N, row, col): // Check col, diagonals for previous rows
    ```
*   **Analysis:** Time roughly `O(N!)` (pruned, much better than `N^N`). Space `O(N)`.

---

## Unit 7: Number Theoretic Algorithms

### 7.1 Number Theoretic Notations, Euclid’s and Extended Euclid’s Algorithms and their Analysis.

*   **Notations:**
    *   `a | b`: `a` divides `b`.
    *   `gcd(a, b)`: Greatest Common Divisor.
    *   `a ≡ b (mod m)`: `m | (a-b)`.
    *   `a⁻¹ mod m`: Modular multiplicative inverse.
*   **Euclid’s Algorithm (GCD):** (Covered in Unit 2.1) Time `O(log(min(a,b)))`.
*   **Extended Euclid’s Algorithm:**
    *   Finds `x, y` such that `ax + by = gcd(a, b)`. Used for modular inverse.
    ```
    function extended_euclid(a, b): // returns [gcd, x, y]
        if b == 0: return [a, 1, 0]
        [gcd_val, x1, y1] = extended_euclid(b, a mod b)
        x = y1
        y = x1 - (a // b) * y1
        return [gcd_val, x, y]
    ```
    *   **Analysis:** Time `O(log(min(a,b)))`. Space `O(log(min(a,b)))` (recursive).

### 7.2 Solving Modular Linear Equations, Chinese Remainder Theorem, Primality Testing: Miller-Rabin Randomized Primality Test and their Analysis

#### Solving Modular Linear Equations `ax ≡ b (mod m)`
1.  `g = gcd(a, m)`. If `b` not divisible by `g`, no solutions.
2.  Else, `g` solutions. Divide by `g`: `(a/g)x ≡ (b/g) (mod m/g)`.
3.  Let `a'=a/g, b'=b/g, m'=m/g`. Solve `a'x ≡ b' (mod m')`. Find `(a')⁻¹ mod m'`.
4.  `x₀ = ((a')⁻¹ * b') mod m'`.
5.  Solutions: `x = x₀ + k*m'` for `k = 0..g-1`.
*   **Analysis:** Dominated by Extended Euclid: `O(log(min(a,m)))`.

#### Chinese Remainder Theorem (CRT)
*   **Problem:** Solve system `x ≡ a_i (mod m_i)` for pairwise coprime `m_i`.
*   Unique solution modulo `M = m₁*...*m_k`.
*   Solution: `x = (Σ a_i * M_i * y_i) mod M`, where `M_i = M/m_i` and `y_i = (M_i)⁻¹ mod m_i`.
*   **Analysis:** `O(k * log M + (length of M)^2)` or `O(k * log(max m_i))`.

#### Primality Testing: Miller-Rabin Randomized Primality Test
*   Probabilistic (Monte Carlo). If "composite", definitely composite. If "prime", high probability.
*   Basis: Fermat's Little Theorem (`a^(n-1) ≡ 1 (mod n)` if `n` prime) + property `x² ≡ 1 (mod n) => x ≡ ±1 (mod n)`.
*   **Algorithm (1 iteration for odd `n > 2`):**
    1. Write `n-1 = 2^s * d` (`d` is odd).
    2. Pick random `a` in `[2, n-2]`.
    3. `x = a^d mod n`.
    4. If `x == 1` or `x == n-1`, maybe prime (`continue` outer loop).
    5. For `r` from `1` to `s-1`:
        `x = x² mod n`.
        If `x == n-1`, maybe prime (`break` and `continue` outer).
        If `x == 1`, composite (`return "composite"`).
    6. If loop finishes and `x != n-1`, composite (`return "composite"`).
*   If `k` iterations pass: `return "probably prime"`.
*   **Analysis:** Time `O(k * (log n)³)`. Error prob `≤ (1/4)^k`.

---

## Unit 8: NP Completeness

### 8.1 Tractable and Intractable Problems, Concept of Polynomial Time and Super Polynomial Time Complexity

*   **Tractable:** Solvable in polynomial time (`O(n^k)`). E.g., Sorting.
*   **Intractable:** No known polynomial-time algorithm. E.g., TSP.
*   **Super Polynomial Time:** Faster growth than any polynomial (`O(2ⁿ)`, `O(n!)`).

### 8.2 Complexity Classes: P, NP, NP-Hard and NP-Complete

(For decision problems - yes/no answer)

*   **Class P:** Solvable by deterministic algorithm in polynomial time.
*   **Class NP (Nondeterministic Polynomial Time):**
    *   "Yes" solution verifiable by deterministic algorithm in polynomial time.
    *   P ⊆ NP. Open question: P = NP? (Likely P ≠ NP).
*   **Class NP-Hard:**
    *   Problem `H` is NP-Hard if every problem `L ∈ NP` polynomially reduces to `H` (`L ≤ₚ H`).
    *   At least as hard as hardest NP problems. Need not be in NP.
*   **Class NP-Complete (NPC):**
    *   Problem `C` is NP-Complete if:
        1. `C ∈ NP`.
        2. `C` is NP-Hard.
    *   Hardest problems in NP. If one is P, then P=NP.
    *   Examples: SAT, 3-SAT, Subset Sum, Hamiltonian Cycle, Vertex Cover.

Diagram (assuming P ≠ NP):

+-----------------------------------+
| NP-Hard                           |
|   +---------------------------+   |
|   | NP-Complete (NPC)         |   |
|   |      +---------------+    |   |
|   |      | P             |    |   |
|   |      +---------------+    |   |
|   +---------------------------+   |
+-----------------------------------+
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
### 8.3 NP Complete Problems, NP Completeness and Reducibility, Cooks Theorem, Proofs of NP Completeness (CNF SAT, Vertex Cover and Subset Sum)

*   **Reducibility (`L₁ ≤ₚ L₂`):** Poly-time transform of `L₁` instance to `L₂` instance, preserving "yes/no".
*   **Cook's Theorem (Cook-Levin):** Boolean Satisfiability (SAT) is NP-Complete. First NPC problem.
    *   Proof idea: Simulate any NP Turing machine with a Boolean formula.

*   **Proving NP-Completeness for Problem X:**
    1.  Show `X ∈ NP` (poly-time verification).
    2.  Show `X` is NP-Hard (reduce known NPC problem `Y` to `X`, i.e., `Y ≤ₚ X`).

*   **NP-Completeness Proof Sketches:**

    *   **CNF-SAT:** (SAT for Conjunctive Normal Form formulas)
        1. In NP: Verify assignment.
        2. NP-Hard: `SAT ≤ₚ CNF-SAT` (convert any formula to CNF poly-time).

    *   **Vertex Cover (VC):** (Graph `G`, `k`. VC of size `≤ k`?)
        1. In NP: Verify `k` vertices cover all edges.
        2. NP-Hard: `3-SAT ≤ₚ VC`. Construct graph from 3-SAT formula with variable gadgets (2 vertices + edge) and clause gadgets (triangle of 3 vertices). Set `k` appropriately.

    *   **Subset Sum (Decision):** (Set `S`, target `T`. Subset sums to `T`?)
        1. In NP: Verify subset sum.
        2. NP-Hard: `3-SAT ≤ₚ Subset Sum` or `VC ≤ₚ Subset Sum`. Involves constructing large numbers where bit positions correspond to elements/constraints.

### 8.4 Approximation Algorithms: Concept, Vertex Cover Problem, Subset Sum Problem

*   For NP-Hard optimization problems. Poly-time, finds "good" but not necessarily optimal.
*   **Approximation Ratio `ρ`:**
    *   Min problem: `C_approx ≤ ρ * C_opt` (`ρ ≥ 1`).
    *   Max problem: `C_approx ≥ ρ * C_opt` (`ρ ≤ 1`, or use `r = C_opt / C_approx ≥ 1`).
*   **PTAS/FPTAS:** `(1+ε)` or `(1-ε)` approx. FPTAS poly-time in input size AND `1/ε`.

*   **Vertex Cover Problem (Approximation):**
    *   **2-Approximation Algorithm:**
        1. While edges remain: Pick an edge `(u,v)`, add `u,v` to cover, remove incident edges.
    *   Analysis: `|C_approx| = 2 * |Matching_edges| ≤ 2 * |C_opt|`.

*   **Subset Sum Problem (Optimization Approximation):** (Maximize sum `≤ T`)
    *   **FPTAS Sketch:** Maintain list `L_i` of achievable sums using first `i` items. Trim list at each step: if `y_k > y_{last} * (1 + ε/(2n))`, add `y_k`. Keeps list small.
    *   Returns sum `S'` where `S' ≥ (1-ε)S*`. Time `O(n²/ε)` (approx).

---

# Laboratory Works:
This course can be learnt in effective way only if we give focus is given in practical aspects of algorithms and techniques discussed in class. Therefore student should be able to implement the algorithms and analyze their behavior.

For the laboratory work, students should implement the following algorithms in C/ C++ and perform their analysis for time and space complexity.

1.  Basic iterative algorithms GCD algorithm, Fibonacci Sequences, Sequential and Binary Search.
2.  Basic iterative sorting algorithms: Bubble Sort, selection Sort, Insertion Sort.
3.  Binary Search with Divide and conquer approach.
4.  Merge Sort, Heap sort, Quick Sort, Randomized Quick Sort.
5.  Selection Problem with divide and Conquer approach
6.  Fractional Knapsack Problem, Job sequencing with deadline, Kruskal’s algorithm, Prims algorithm, Dijkstra’s Algorithm
7.  Implement the dynamic programming algorithms. (E.g., MCM, LCS, 0/1 Knapsack)
8.  Algorithms using Backtracking approach. (E.g., N-Queens, Subset Sum)
9.  Implement approximationAlgorithm. (E.g., Approx Vertex Cover)

---

# Text Books:
1.  Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest and Clifford Stein, “Introduction to algorithms”, Third Edition.. The MIT Press, 2009.
2.  Ellis Horowitz, SartajSahni, SanguthevarRajasekiaran, “Computer Algorithms”, Second Edition, Silicon Press, 2007.
3.  Kleinberg, Jon, and Eva Tardos, “Algorithm Design”, Addison-Wesley, First Edition, 2005.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

This structure should give you a very good starting point for your Word document. Remember to apply Word's styling features for the best look and for features like an automatic Table of Contents.