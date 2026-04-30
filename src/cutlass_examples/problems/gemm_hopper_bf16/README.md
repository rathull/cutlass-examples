# Kernels

## CUDA

### v0: Naive

Each thread computes a single element of the output matrix `C`. The thread assigned to `C[i, j]` iterates over `A[i, k]` and `B^T[j, k]` for all `k` and computes a dot product.

### v1: Tiling, SMEM, Loop Unrolling

This kernel uses both SMEM tiling and register tiling to compute the output matrix. Each CTA first loads a tile of `A` and `B^T` into SMEM cooperatively using wide vector loads. Then, each thread computes a tile of the output matrix `C` in registers using the SMEM input tiles, and writes the result to HBM.

### v2: Asynchronous Kernel with TMA and WGMMA

The design space for asynchronous kernels with TMA and WGMMA includes deciding:
1. Tile shape `(BM, BN, BK)`. WGMMA takes on shapes `m64nNk16` for `n ∈ {8, 16, 32, 64, 128, 256}`.
    1. `BK` is a multiple of 16. Within a single stage, a TMA issues a load for `BM*BK` and `BN*BK` elements. The consumer will use `BK / 16` WGMMA instructions back-to-back, accumulating into the same registers while walking K, to consumer these inputs. Since TMA wants big, continuous transfers, we typically want something close to `BK = 64`. `num_stages` is bound by SMEM, so if each stage is smaller, we also need more stages since we're consumer them much faster, and the producer will have to issue many more TMAs to keep up, leading to more overhead (from TMAs, WGMMAs, mbarrier transactions, etc.). Larger `BK` amortizes this. If `BK` is too large then a single stage may exceed what the consumer can drain before the producer wraps around the ring buffer, or `num_stages` will be too small to hide latency. 
    2. `BM = 64 * num_consumer_groups` and `BN = N * num_consumer_groups`
2. Consumer warpgroups (1-8) determines `BM`, `BN`, and register pressure
    1. More consumer warpgroups improves the arithmetic intensity for B, since B is shared across WGs within a stage (and A is partitioned by row). So ideally we have large N and large `num_consumer_groups`.
3. Warp specialization: producer/consumer vs. unified warps, combined with `setmaxnreg` to reallocate registers towards consumers
    1. The benefit of WS would be register splitting via `setmaxnreg`, no context-switching in the consumer (so the consumer can be a tight WGMMA loop and producer can be a tight TMA issue loop so each side has cleaner ILP), and the producer can race ahead more aggressively and not block on `producer_acquire`. 
    2. This is a bigger deal if we're compute-bound, if we're memory-bound then we're at HBM peak
4. `num_stages`: number of K-tile buffers in SMEM, bounded by SMEM capacity
5. Persistent vs. grid-launched
6. Cluster shape: with clusters of 2+, TMA multicase can broadcast an operand across CTAs sharing an M or N tile and halve HBM traffic for that operand
7. Tile scheduler/threadblock swizzling: row-major, threadblock-swizzles, Hilbert, Stream-k
8. Stream-k vs. Split-K for skinny problems with too few tiles to fill 132 SMs
9. Epilogue fushion
10. SMEM layout/swizzle for A and B

For this first kernel, we'll implement pipelined TMA and WGMMA with the following hyperparameters:
1. `BM`: 64, 128, 256 (64 * `num_consumer_groups`)
2. `BN`: 8, 16, 32, 64, 128, 256
3. `BK`: 16, 32, 64, 128, 256
4. `num_stages`: 2, 4, 8, 16, 32

We can derive the occupancy constraints from the above:
- The grid size is `(ceil(N / BN), ceil(M / BM))`, so the wave count is `M * N / (132 * BN * BM)`.
- Each CTA has `(BM/64) * 32 * 4 = 2 * BM` consumer threads + `32` producer threads for a total of `2 * BM + 32` threads.
    - So we need `BM ∈ {64, 128, 256, 512}`.
- Max TMA laod size is 256 elements in any dimension
    - So we need `BM ∈ {64, 128, 256}`
- SMEM per CTA is `num_stages * (BM * BK + BN * BK)` BF16s = `2 * num_stages * (BM * BK + BN * BK)` bytes.
    - So we need `2 * num_stages * (BM * BK + BN * BK) <= 227` KB.
- Accumulator registers per CTA is `(BM/64) * (BN/8) * 4 = BM * BN / 16` FP32s = `BM * BN / 4` bytes.
    - So we need `BM * BN / 4 <= 256` KB.

With `BM=128`, `BN=64`, `BK=64`, `num_stages=4`, we have:
- SMEM per CTA is 98,304 bytes, so 2 CTAs/SM
- Accumulator registers per CTA is 2,048 bytes, so this is not a limiation

#### Registers and `setmaxnreg`

For warp specialization, we need to repartition registers towards consumers.

Typically, we use R_p = 40 and R_c = 232 (occasionally 240)

Max registers per thread is 255 registers, registers per SM is 65,536 registers.
- We need BM * BN registers per CTA for FP32 accumulators.
    - Since a CTA has 2 * BM + 32 threads, this is (BM * BN) / (2 * BM) = BN / 2 registers per consumer thread
- Assuming occupancy of 2 CTAs/SM, limit is 2 * BM * BN registers active per SM < 65,536 regs/SM

Thread limit: 2 * BN + other <= 255
CTA limit:    BM * BN + other <= 32,768

So realistically, BN = 256 is the max we'd run, any anyways this is the max WGMMA N supported.

C1: 1 CTA/SM
- 32 * R_p + 128 * ncg * R_c <= 65,536
- With ncg = 1, we get 32 * 40 + 128 * 232 = 30,720 registers active per SM
- With ncg = 2, we get 32 * 40 + 128 * 2 * 232 = 61,440 registers active per SM, which is tight
    - Definitely only get 1 CTA/SM
- With ncg = 4, we need to drop R_c to ~160, which is too tight if BN >= 128.
    - TODO: what to do in this case? Do we just use ncg=1 and issue 4 WGMMAs from this WG? Or do we use ncg=2 and issue 2 WGMMAs from this WG? Or something else? And would either of these possilby be beter than just using more CTAs with smaller BM?
C2: 2 CTAs/SM
- 2 * (32 * R_p + 128 * ncg * R_c) <= 65,536
- With ncg = 1, we get 2 * (32 * 40 + 128 * 232) = 61,440 registers active per SM, which is tight

TODO: how do the above change if occupancy is 2 CTAs vs. 1 CTA per SM?



For auto kernel:
```bash
claude --resume eb3156f9-5098-4c24-b174-65a42cab257a
```