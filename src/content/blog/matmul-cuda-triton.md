---
title: "gpu-01: Matmul with CUDA and Triton"
description: ""
pubDate: "Jul 11 2026"
updatedDate: "Jul 11 2026"
heroImage: ""
---

Hello Everyone, this is a blog series where I will note down my learnings while working with CUDA and Triton.
This is the first post in the series.

## Understanding the GPU
I will not go into detail about the GPU here, but I will give a high-level overview of how it works. For more details, you can refer to the [Understanding the GPU](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html).

So here we go!

blah blah blah !

## understanding Matmul
Matmul is the matrix multiplication operation<span class="sidenote-ref"></span><span class="sidenote">Given A (m×n) and B (n×p), matmul produces C (m×p) where each element is a dot product of a row of A and a column of B.</span> that we will be working with in this series. It is a fundamental operation in linear algebra and is used in many machine learning algorithms.

## Matmul on the GPU
Matmul on the GPU can be accelerated using CUDA and Triton<span class="sidenote-ref"></span><span class="sidenote">CUDA gives you low-level control over threads and memory; Triton is a higher-level language that compiles down to efficient GPU kernels without hand-managing most of that.</span>. In this post, we will focus on using Triton to implement Matmul on the GPU. I will also show you the equivalent CUDA code for comparison.
