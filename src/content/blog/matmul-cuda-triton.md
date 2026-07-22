---
title: "gpu-01: Matmul with CUDA and Triton"
description: ""
pubDate: "Jul 11 2026"
updatedDate: "Jul 11 2026"
heroImage: ""
---

This is the first post in a series where I note down what I learn while working with CUDA and Triton. We'll start with the "hello world" of GPU programming: matrix multiplication.

## Why the GPU is different

I won't go deep into GPU architecture here — for that, the [modern GPU programming notes](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html) are a much better reference. But there's one idea I need up front, because everything else in this post depends on it.

A CPU has a handful of very fast cores. It's built to run one thing (or a few things) as quickly as possible. A GPU is the opposite: thousands of small, slower cores that all run at once. It's bad at doing one thing fast and great at doing the same thing to a lot of data at the same time.

Matrix multiplication is exactly that "same thing, lots of data" shape, which is why it's the perfect first example.

## What matmul actually is

Matmul is matrix multiplication<span class="sidenote-ref"></span><span class="sidenote">Given A (m×n) and B (n×p), matmul produces C (m×p), where each element `C[i][j]` is the dot product of row `i` of A and column `j` of B.</span> — a fundamental operation in linear algebra, and the workhorse behind almost every machine learning model. Here it is as a plain Python loop:

```python
def matmul(a, b):
    # a is (M x K), b is (K x N), result is (M x N)
    result = [[0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):            # each row of a
        for j in range(len(b[0])):     # each column of b
            for k in range(len(a[0])):  # dot product over the shared dim
                result[i][j] += a[i][k] * b[k][j]
    return result
```

This is the naive version, and it runs in O(n³) time — for every one of the `M × N` output elements, we do a dot product of length `K`. On a CPU those three loops run one after another, one element at a time. You can make this a lot faster on the CPU too (NumPy's `matmul`, cache blocking, SIMD); the [ppc notes](https://ppc.cs.aalto.fi/ch2/) walk you through squeezing real performance out of C++ on the CPU, and they're worth reading.

But here's the thing that matters for us: look at that triple loop. Every output element `C[i][j]` is computed completely independently of every other one. Nothing in `C[0][0]` depends on `C[0][1]`. That independence is the whole reason matmul maps so well onto a GPU — instead of computing the elements one after another, we can compute all of them at once, one thread per element.

That's the jump we're about to make. Same math, but instead of looping over `i` and `j`, we hand each `(i, j)` to its own thread.

## Matmul on the GPU
Matmul on the GPU can be accelerated using CUDA and Triton<span class="sidenote-ref"></span><span class="sidenote">CUDA gives you low-level control over threads and memory; Triton is a higher-level language that compiles down to efficient GPU kernels without hand-managing most of that.</span>. Let's start with CUDA, because it forces you to think about how the work is actually split across threads. Then we'll write the same thing in Triton and see how much of that bookkeeping goes away.

## The CUDA version

Remember the naive CPU loop from earlier — three nested loops walking over `i`, `j`, `k`. On the GPU we don't loop over `i` and `j` at all. Instead we launch one thread for every output element `c[i][j]`, and each thread does *only* the inner `k` loop for its own element.

So a thread's whole job is: "figure out which `(row, col)` of `C` I am responsible for, then compute the dot product of that row of `A` and that column of `B`."

Here is the kernel:

```cpp
__global__ void matrixMultiplication(int *a, int *b, int *c, int N) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    int sum = 0;
    for (int i = 0; i < N; i++) {
      sum += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
  }
}
```

A few things to notice here.

The `row` and `col` lines are how a thread finds out who it is<span class="sidenote-ref"></span><span class="sidenote">Threads are grouped into blocks, and blocks into a grid. `threadIdx` is the thread's position inside its block, `blockIdx` is the block's position inside the grid, and `blockDim` is how many threads per block. Multiply and add and you get a unique global coordinate.</span>. Every thread runs the exact same code, but because each one gets a different `(row, col)`, they all end up working on different output elements.

We index the matrices with `row * N + i` instead of `a[row][i]` because the matrix is stored as one flat array in memory — row after row. So element `(r, c)` lives at position `r * N + c`. That's why `a[row * N + i]` walks along a row and `b[i * N + col]` walks down a column.

The `if (row < N && col < N)` guard matters because we usually launch a few more threads than we strictly need (more on that below), and we don't want those extra threads writing out of bounds.

Now the launch side:

```cpp
int main() {
  int *a, *b, *c;
  int N = 1 << 10;                       // 1024
  size_t bytes = (N * N * sizeof(int));
  cudaMallocManaged(&a, bytes);          // unified memory: CPU and GPU share it
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);
  initMatrix(a, N);
  initMatrix(b, N);

  int threads = 16;
  int blocks = (N + threads - 1) / threads;  // ceil(N / threads)
  dim3 THREADS(threads, threads);            // 16x16 = 256 threads per block
  dim3 BLOCKS(blocks, blocks);               // enough blocks to cover N x N

  matrixMultiplication<<<BLOCKS, THREADS>>>(a, b, c, N);
  cudaDeviceSynchronize();
}
```

The important idea is the grid. We want one thread per output element, and `C` is `N x N`. So we make each block a `16 x 16` square of threads, and then we need `ceil(N / 16)` blocks in each dimension to cover the whole matrix. The `(N + threads - 1) / threads` is just integer ceiling division<span class="sidenote-ref"></span><span class="sidenote">If `N` isn't a multiple of 16, this rounds up so we always have *at least* enough threads. That's exactly why the kernel needs the `row < N && col < N` bounds check — the last blocks overhang the matrix.</span>.

`cudaMallocManaged` gives us unified memory, so we don't have to manually `cudaMemcpy` data back and forth — the driver moves pages between CPU and GPU for us. And `cudaDeviceSynchronize()` at the end makes the CPU wait for the GPU to actually finish, since kernel launches are asynchronous.

This works, and it's a fine mental model. But notice every thread reads its entire row of `A` and column of `B` straight from global memory, and neighbouring threads re-read a lot of the same values. That's slow. The usual fix is *tiling* — load a small block of `A` and `B` into fast shared memory once, and let all the threads in the block reuse it. Writing that by hand in CUDA is fiddly. This is exactly where Triton starts to shine.

## The Triton version

Triton lets us work at the level of *tiles* (small blocks of the matrix) instead of individual threads. We don't say "I am thread `(row, col)`" — we say "I am the block responsible for this `BLOCK_SIZE x BLOCK_SIZE` tile of `C`", and Triton handles the threads underneath.

Here's the kernel:

```python
@triton.jit
def _mul(_ptr_a, _ptr_b, _ptr_c, N, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    row_start = pid_y * BLOCK_SIZE
    col_start = pid_x * BLOCK_SIZE
    rows = row_start + tl.arange(0, BLOCK_SIZE)
    cols = col_start + tl.arange(0, BLOCK_SIZE)

    # accumulator holds this output tile of C while we sum over k
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    k_idx = tl.arange(0, BLOCK_SIZE)

    a_offsets = rows[:, None] * N + k_idx[None, :]
    b_offsets = k_idx[:, None] * N + cols[None, :]
    c_offsets = rows[:, None] * N + cols[None, :]

    for k in range(0, N, BLOCK_SIZE):
        a_tile = tl.load(_ptr_a + a_offsets)
        b_tile = tl.load(_ptr_b + b_offsets)
        acc = tl.dot(a_tile, b_tile, acc)
        a_offsets = a_offsets + BLOCK_SIZE       # slide right across A
        b_offsets = b_offsets + BLOCK_SIZE * N   # slide down through B

    acc = acc.to(tl.float16)
    tl.store(_ptr_c + c_offsets, acc)
```

The `program_id` is the Triton equivalent of `blockIdx` — it tells this program instance which tile it owns. `pid_x` and `pid_y` give us the tile's column and row position, and multiplying by `BLOCK_SIZE` gives us the actual starting row and column in the matrix.

Now the interesting part is the offsets. In CUDA a single thread computed a single element. Here we're computing a whole `BLOCK_SIZE x BLOCK_SIZE` tile at once, so instead of a single index we build a 2D grid of indices<span class="sidenote-ref"></span><span class="sidenote">`rows[:, None]` makes a column vector and `k_idx[None, :]` makes a row vector; adding them broadcasts into a full `BLOCK_SIZE x BLOCK_SIZE` matrix of memory offsets. Same flat `r * N + c` indexing as the CUDA version, just done for a whole tile at a time.</span>. `a_offsets` picks out a tile of `A`, `b_offsets` a tile of `B`, and `c_offsets` where the result goes in `C`.

The loop is the `k` loop, but tiled. Instead of stepping one element at a time, we step `BLOCK_SIZE` columns of `A` and `BLOCK_SIZE` rows of `B` per iteration. Each iteration we load a tile of `A` and a tile of `B`, do a small matrix multiply with `tl.dot`, and add it into the accumulator. `tl.dot` is the big one — it's what maps onto the GPU's tensor cores<span class="sidenote-ref"></span><span class="sidenote">Tensor cores are dedicated hardware units that do small matrix multiplies in a single operation, much faster than doing the multiply-adds one by one. Feeding them fp16 tiles is the fast path.</span>, so this one call replaces the whole inner scalar loop from the CUDA kernel.

We keep the accumulator in `float32` even though the inputs are `float16`. Summing `N` products in fp16 would lose too much precision, so we accumulate wide and only cast back down to fp16 right before storing.

After the loop, `a_offsets` slides right (`+ BLOCK_SIZE`) because we're moving along the columns of `A`, and `b_offsets` slides down (`+ BLOCK_SIZE * N`) because we're moving down the rows of `B`. That's the tiled walk over `k`.

Now the host side that launches it:

```python
def mul(a, b, c, N, BLOCK_SIZE=32):
    # a 2D grid of blocks. NOTE: triton.cdiv here (host side), not tl.cdiv
    # (tl.* only works inside a @triton.jit kernel).
    grid = (
        triton.cdiv(N, BLOCK_SIZE),
        triton.cdiv(N, BLOCK_SIZE),
    )
    _mul[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
```

This is the same idea as the CUDA `dim3 BLOCKS` — a 2D grid with `ceil(N / BLOCK_SIZE)` tiles in each direction. `triton.cdiv` is just ceiling division; the naming trap is that inside the kernel you'd use `tl.cdiv`, but out here on the host it's `triton.cdiv`.

## Checking it works

Before trusting any of this, compare against PyTorch's own matmul:

```python
def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    N = 1 << 10  # 1024 x 1024, a multiple of BLOCK_SIZE so no masks needed yet
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(N, N, device="cuda", dtype=torch.float16)

    mul(a, b, c, N)                 # warmup / JIT compile
    ref = torch.matmul(a, b)
    torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
    print("correctness OK")
```

Two things worth calling out. First, I picked `N = 1024`, which is a clean multiple of `BLOCK_SIZE = 32`, so every tile lands fully inside the matrix and we don't need boundary masks yet<span class="sidenote-ref"></span><span class="sidenote">If `N` weren't a multiple of `BLOCK_SIZE`, the edge tiles would read past the end of the matrix and we'd need a mask in `tl.load`/`tl.store` to skip those out-of-range elements — the Triton equivalent of the CUDA `row < N && col < N` guard. A problem for a later post.</span>. Second, the tolerance is loose on purpose — fp16 only has about three decimal digits of precision, and summing 1024 products drifts, so we can't expect an exact match.

The first `mul` call also doubles as a warmup, because Triton compiles the kernel just-in-time on first use. You don't want that compile time landing inside your benchmark.

## Benchmarking

```python
    avg = 0.0
    iters = 5
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        mul(a, b, c, N)
        torch.cuda.synchronize()
        end = time.time()
        avg += end - start
        print(f"run {i} took {end - start:.6f} seconds")
    avg /= iters

    tflops = 2 * N * N * N / avg / 1e12
    print(f"avg: {avg:.6f} s   ~{tflops:.1f} TFLOP/s")
```

The `torch.cuda.synchronize()` calls are the whole trick to timing GPU code. GPU work is asynchronous — the launch returns immediately while the GPU is still busy. If you time without synchronizing, you're just measuring how fast Python can queue up work, not how long the work takes. Synchronizing before `start` makes sure nothing is left over, and synchronizing before `end` makes sure the kernel has actually finished.

The `2 * N^3` in the TFLOP/s line comes from the cost of matmul: each of the `N * N` output elements needs `N` multiply-adds, and a multiply-add counts as two floating point operations. Divide by the time and by `1e12` and you get trillions of FLOPs per second, which is the number everyone quotes for matmul performance.

## Wrapping up

So that's the same operation twice: once in CUDA where we manage individual threads by hand, and once in Triton where we think in tiles and let `tl.dot` hit the tensor cores for us. The Triton version is shorter, and it's already on the fast fp16 path — but we cheated a little by only handling sizes that are multiples of the block size. Handling ragged sizes with masks, and then actually tuning `BLOCK_SIZE` for real speed, is where the next posts go.
