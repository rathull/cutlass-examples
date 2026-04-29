// =============================================================================
// cuda_claude.cu — Hopper (sm_90) BF16 GEMM with all the right techniques.
//
// Layout:        NT — A is M×K row-major; "B" arg is N×K row-major (logical Bᵀ).
//                C = alpha · A · Bᵀ + beta · C.
//
// Tile:          CTA = BM × BN × BK   (default 128 × 192 × 64)
// Warp groups:   1 producer wg (128 threads, 1 active warp issues TMAs)
//                2 consumer wgs (128 threads each) split BM along M.
//                Total 384 threads / CTA.
// Loads:         TMA (cp.async.bulk.tensor.2d) into 128B-swizzled SMEM, signalled
//                by mbarrier transactions. Multi-stage producer/consumer ring.
// Math:          wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 with
//                FP32 accumulators, async commit_group / wait_group pipelining.
// Schedule:      Persistent CTAs (one per SM). Output tile order is rasterised
//                in a 4-tile super-block to maximise L2 reuse of A/B rows/cols.
// Epilogue:      Per-thread vector stores to gmem; bf162 packed pairs.
//
// Numbers below assume the defaults above; the static_asserts catch mismatches.
// =============================================================================

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#include <cstdint>

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 64
#endif
#ifndef NUM_STAGES
#define NUM_STAGES 4
#endif
#ifndef CLUSTER_M
#define CLUSTER_M 1
#endif
#ifndef CLUSTER_N
#define CLUSTER_N 1
#endif

#ifndef ENTRYPOINT
#define ENTRYPOINT cuda_claude
#endif
#ifndef KERNEL_NAME
#define KERNEL_NAME cuda_claude_kernel
#endif

namespace claude {

// ----- Compile-time constants -------------------------------------------------
constexpr int kBM = BM;
constexpr int kBN = BN;
constexpr int kBK = BK;
constexpr int kStages = NUM_STAGES;

constexpr int kWgmmaM = 64;
constexpr int kWgmmaN = kBN;
constexpr int kWgmmaK = 16;

constexpr int kConsumerWgs = 2;
constexpr int kProducerWgs = 1;
constexpr int kThreadsPerWg = 128;
constexpr int kNumThreads = (kConsumerWgs + kProducerWgs) * kThreadsPerWg;

constexpr int kKInner = kBK / kWgmmaK;                     // wgmma issues per stage per wg
constexpr int kConsumerRows = kBM / kConsumerWgs;          // rows owned by one consumer wg
static_assert(kConsumerRows == kWgmmaM, "Each consumer wg owns one wgmma m-tile.");

constexpr int kAccPerThread = (kWgmmaM * kWgmmaN) / 128;   // fp32 fragment size per consumer thread
static_assert(kBN % 8 == 0 && kBN >= 8 && kBN <= 256, "Invalid BN for wgmma.m64nNk16.");
static_assert(kBK % kWgmmaK == 0, "BK must be a multiple of wgmma K = 16.");
static_assert(kBK % 32 == 0, "BK*sizeof(bf16) must be a multiple of 64 bytes (swizzle requirement).");
static_assert(kStages >= 2, "Need at least 2 pipeline stages.");
static_assert(kBM % kConsumerWgs == 0 && (kBM / kConsumerWgs) == kWgmmaM,
              "BM must split evenly into kConsumerWgs wgmma M-tiles.");

// ----- 128B swizzle SMEM tiles ----------------------------------------------
//
// For 128B swizzle (CU_TENSOR_MAP_SWIZZLE_128B), the SMEM tile is laid out so
// each contiguous 128 byte (= 64 bf16) row fits a wgmma core-matrix row, and
// rows are XOR-swizzled to be conflict-free.  We reflect that via simple [N][K]
// arrays with K = multiple of 64 bf16; LBO/SBO in the matrix descriptor encode
// the canonical 128B-swizzle pattern.

constexpr int kBKBytes = kBK * static_cast<int>(sizeof(__nv_bfloat16));    // 128 for BK=64
constexpr int kATileBytes  = kBM * kBKBytes;                                // 128 * 128 = 16384
constexpr int kBTileBytes  = kBN * kBKBytes;                                // 192 * 128 = 24576
constexpr int kStageBytes  = kATileBytes + kBTileBytes;                     // 40960
constexpr int kSmemBytes   = kStages * kStageBytes
                           + kStages * 2 * sizeof(uint64_t)                 // mbarriers
                           + 16;                                            // align headroom

// Layout in dynamic shared memory:
//   [0 .. kStages * kATileBytes)               sA tiles (one per stage)
//   [.. + kStages * kBTileBytes)               sBT tiles
//   [.. + kStages * sizeof(uint64_t))          full[stage] mbarriers
//   [.. + kStages * sizeof(uint64_t))          empty[stage] mbarriers

// ----- Inline-PTX primitives --------------------------------------------------

__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ----- mbarrier ops ----------------------------------------------------------

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t addr = to_smem(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    uint32_t addr = to_smem(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t addr = to_smem(bar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(addr), "r"(bytes)
    );
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t addr = to_smem(bar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT_%=: \n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra DONE_%=;\n"
        "bra LAB_WAIT_%=;\n"
        "DONE_%=: \n"
        "}\n"
        :: "r"(addr), "r"(phase)
    );
}

__device__ __forceinline__ void fence_async_proxy() {
    asm volatile("fence.proxy.async.shared::cta;\n" :::);
}

// ----- TMA load 2D -----------------------------------------------------------

__device__ __forceinline__ void tma_load_2d(
    void* smem_dst, const CUtensorMap* desc, int crd0, int crd1, uint64_t* mbar
) {
    uint32_t smem = to_smem(smem_dst);
    uint32_t bar  = to_smem(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem), "l"(desc), "r"(crd0), "r"(crd1), "r"(bar)
        : "memory"
    );
}

__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* desc, int crd0, int crd1) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];\n"
        :: "l"(desc), "r"(crd0), "r"(crd1)
        : "memory"
    );
}

// ----- WGMMA descriptor (K-major BF16, 128B swizzle) -------------------------
//
// Descriptor bit layout (64-bit):
//   [0..13]  start_address      = smem_byte_offset >> 4    (4LSB dropped, u128 units)
//   [16..29] leading_byte_offset = stride<1,0> in u128_t units
//   [32..45] stride_byte_offset  = stride<0,1> in u128_t units
//   [49..51] base_offset        = 0 for our aligned tiles
//   [62..63] layout_type        = 0=INTERLEAVE, 1=B128, 2=B64, 3=B32
//
// CUTLASS's canonical K-major B128 layout (cute/atom/mma_traits_sm90_gmma.hpp):
//     LayoutType::B128 : Swizzle<3,4,3> o smem_ptr o ((8, n), 2) : ((8, SBO), 1)
// in uint128_t (16-byte) units:
//   leading_byte_offset = 1                           (stride between u128 in K)
//   stride_byte_offset  = 8 * BK * 2 / 16 = BK        (stride between 8-row M-blocks)
//
// Our SMEM tile is sA[BM][BK] / sBT[BN][BK] row-major BF16, K-contiguous, with
// BK*sizeof(bf16) = 128 B per row.  TMA writes 128B-swizzled, WGMMA reads it
// the same way via layout_type = 1.

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_byte_addr) {
    constexpr uint64_t lbo_enc = 1;                                      // 16 B (1 u128)
    constexpr uint64_t sbo_enc = static_cast<uint64_t>(kBK);              // BK u128 units
    uint64_t desc = 0;
    desc |= (static_cast<uint64_t>(smem_byte_addr >> 4) & 0x3FFFull);
    desc |= (lbo_enc << 16);
    desc |= (sbo_enc << 32);
    desc |= (static_cast<uint64_t>(1) << 62);                             // B128 swizzle
    return desc;
}

// ----- WGMMA ops -------------------------------------------------------------

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// Per-N specialisations for wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16.
// Each thread holds N/2 fp32 outputs in d[].  trans_a = trans_b = 0 (K-major).
template <int ScaleD>
__device__ __forceinline__ void wgmma_m64nXk16_bf16(
    float* d, uint64_t descA, uint64_t descB
);

#define CLAUDE_WGMMA_REG_LIST_64 \
    " %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  %8,  %9, %10, %11, %12, %13, %14, %15," \
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31," \
    "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47," \
    "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"

#define CLAUDE_WGMMA_OUT_LIST_64 \
    "+f"(d[ 0]), "+f"(d[ 1]), "+f"(d[ 2]), "+f"(d[ 3]), "+f"(d[ 4]), "+f"(d[ 5]), \
    "+f"(d[ 6]), "+f"(d[ 7]), "+f"(d[ 8]), "+f"(d[ 9]), "+f"(d[10]), "+f"(d[11]), \
    "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), \
    "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
    "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), \
    "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), \
    "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), \
    "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), \
    "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), \
    "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), \
    "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])

// m64n128k16  (BN=128, 64 fp32 outputs/thread)
template <>
__device__ __forceinline__ void wgmma_m64nXk16_bf16<1>(
    float* d, uint64_t descA, uint64_t descB
) {
#if BN == 128
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{" CLAUDE_WGMMA_REG_LIST_64 "}, %64, %65, 1, 1, 1, 0, 0;\n"
        : CLAUDE_WGMMA_OUT_LIST_64
        : "l"(descA), "l"(descB)
    );
#elif BN == 192
    // 96 fp32 outputs/thread
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  %8,  %9, %10, %11, %12, %13, %14, %15,"
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
        " %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
        " %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
        " %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
        " %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95},"
        " %96, %97, 1, 1, 1, 0, 0;\n"
        : CLAUDE_WGMMA_OUT_LIST_64,
          "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]),
          "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),
          "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]),
          "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
          "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]),
          "+f"(d[94]), "+f"(d[95])
        : "l"(descA), "l"(descB)
    );
#elif BN == 256
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  %8,  %9, %10, %11, %12, %13, %14, %15,"
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
        " %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
        " %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
        " %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
        " %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95,"
        " %96, %97, %98, %99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,"
        "%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
        "%128, %129, 1, 1, 1, 0, 0;\n"
        : CLAUDE_WGMMA_OUT_LIST_64,
          "+f"(d[ 64]), "+f"(d[ 65]), "+f"(d[ 66]), "+f"(d[ 67]), "+f"(d[ 68]), "+f"(d[ 69]),
          "+f"(d[ 70]), "+f"(d[ 71]), "+f"(d[ 72]), "+f"(d[ 73]), "+f"(d[ 74]), "+f"(d[ 75]),
          "+f"(d[ 76]), "+f"(d[ 77]), "+f"(d[ 78]), "+f"(d[ 79]), "+f"(d[ 80]), "+f"(d[ 81]),
          "+f"(d[ 82]), "+f"(d[ 83]), "+f"(d[ 84]), "+f"(d[ 85]), "+f"(d[ 86]), "+f"(d[ 87]),
          "+f"(d[ 88]), "+f"(d[ 89]), "+f"(d[ 90]), "+f"(d[ 91]), "+f"(d[ 92]), "+f"(d[ 93]),
          "+f"(d[ 94]), "+f"(d[ 95]), "+f"(d[ 96]), "+f"(d[ 97]), "+f"(d[ 98]), "+f"(d[ 99]),
          "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]),
          "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]),
          "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),
          "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(descA), "l"(descB)
    );
#else
#error "Unsupported BN: add a wgmma specialisation for this width."
#endif
}

// ----- setmaxnreg helpers (Hopper register-budget repartitioning) ------------
template <int N>
__device__ __forceinline__ void warpgroup_dec_regs() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}
template <int N>
__device__ __forceinline__ void warpgroup_inc_regs() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}

}  // namespace claude

// =============================================================================
// Kernel
// =============================================================================

extern "C" __global__ __launch_bounds__(claude::kNumThreads, 1)
void KERNEL_NAME(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_bt,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    int num_tiles_m, int num_tiles_n
) {
    using namespace claude;

    extern __shared__ __align__(128) uint8_t smem_raw[];

    // ---- SMEM partition ------------------------------------------------------
    __nv_bfloat16* sA  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sBT = reinterpret_cast<__nv_bfloat16*>(smem_raw + kStages * kATileBytes);
    uint64_t* full_bar  = reinterpret_cast<uint64_t*>(smem_raw + kStages * kStageBytes);
    uint64_t* empty_bar = full_bar + kStages;

    auto sA_stage  = [&](int s) { return sA  + s * (kBM * kBK); };
    auto sBT_stage = [&](int s) { return sBT + s * (kBN * kBK); };

    const int tid     = threadIdx.x;
    const int wg_idx  = tid / kThreadsPerWg;       // 0,1 = consumers, 2 = producer
    const int lane    = tid % 32;
    const int warp_in_wg = (tid % kThreadsPerWg) / 32;

    const bool is_producer = (wg_idx == kConsumerWgs);
    const int  consumer_id = wg_idx;                         // valid only when !is_producer

    // ---- Initialise mbarriers (one thread) -----------------------------------
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < kStages; ++s) {
            // full[s]: 1 producer arrival completes the load (TMA arrives via expect_tx).
            mbarrier_init(&full_bar[s], 1);
            // empty[s]: kConsumerWgs arrivals = consumers signalling stage available again.
            mbarrier_init(&empty_bar[s], kConsumerWgs);
        }
    }
    // All threads must see initialized mbarriers before any async-proxy access.
    fence_async_proxy();
    __syncthreads();

    // ---- Persistent tile schedule -------------------------------------------
    const int total_tiles = num_tiles_m * num_tiles_n;
    const int num_ctas    = gridDim.x;
    constexpr int kSwizzle = 4;   // super-block of 4 N-tiles for L2 reuse

    auto tile_to_mn = [&](int tile_idx, int& tile_m, int& tile_n) {
        const int stride_n = kSwizzle;
        const int blocks_per_super = num_tiles_m * stride_n;
        const int sb_idx = tile_idx / blocks_per_super;
        const int sb_base_n = sb_idx * stride_n;
        const int rem = tile_idx % blocks_per_super;
        const int local_n = rem / num_tiles_m;
        const int local_m = rem % num_tiles_m;
        tile_m = local_m;
        tile_n = sb_base_n + local_n;
        if (tile_n >= num_tiles_n) {
            // Tail: fall back to row-major for the leftovers
            const int linear = tile_idx;
            tile_m = linear % num_tiles_m;
            tile_n = linear / num_tiles_m;
        }
    };

    if (is_producer) {
        // Producer warp group: only the first warp does work; relinquish regs.
        warpgroup_dec_regs<40>();

        if (warp_in_wg == 0) {
            // Producer waits on empty[stage] starting from the (kStages+1)-th issue.
            // Phase rules (see derivation in design notes):
            //   - empty barriers start at parity 0 with no arrivals.
            //   - 1st producer wait at any stage: parity needs to flip 0→1, pass phase=0.
            //   - 2nd: pass phase=1, etc.
            // Skipping the first kStages waits and starting phase=1 (toggled at each
            // ring wrap to 0, 1, 0, ...) makes the first real wait pass phase=0.
            uint32_t phase = 1;
            int      stage = 0;
            int      global_iter = 0;

            for (int tile = blockIdx.x; tile < total_tiles; tile += num_ctas) {
                int tile_m, tile_n;
                tile_to_mn(tile, tile_m, tile_n);
                const int cta_m = tile_m * kBM;
                const int cta_n = tile_n * kBN;

                const int num_k_tiles = (K + kBK - 1) / kBK;
                for (int kt = 0; kt < num_k_tiles; ++kt) {
                    if (global_iter >= kStages) {
                        mbarrier_wait(&empty_bar[stage], phase);
                    }

                    if (lane == 0) {
                        const uint32_t bytes = static_cast<uint32_t>(kATileBytes + kBTileBytes);
                        mbarrier_arrive_expect_tx(&full_bar[stage], bytes);
                        tma_load_2d(sA_stage(stage), &tma_a,
                                    /*crd0(K)=*/ kt * kBK, /*crd1(M)=*/ cta_m, &full_bar[stage]);
                        tma_load_2d(sBT_stage(stage), &tma_bt,
                                    /*crd0(K)=*/ kt * kBK, /*crd1(N)=*/ cta_n, &full_bar[stage]);
                    }

                    ++global_iter;
                    stage += 1;
                    if (stage == kStages) {
                        stage = 0;
                        phase ^= 1;
                    }
                }
            }
        }
        return;
    }

    // ---- Consumer ------------------------------------------------------------
    warpgroup_inc_regs<232>();

    // Each consumer wg owns kWgmmaM = 64 rows.
    const int wg_m_off  = consumer_id * kWgmmaM;     // 0 or 64
    // Per-thread D fragment.
    float acc[kAccPerThread];

    // Decompose lane index for the m64nNk16 fragment layout:
    //   warp_id within wg: tid_in_wg / 32
    //   lane_in_warp:      tid_in_wg % 32
    //   row_a in 64-tile = warp_id*16 + lane_in_warp/4
    //   row_b           = row_a + 8
    //   col-stripe c    = lane_in_warp % 4         (cols 8k+2c, 8k+2c+1)
    const int tid_in_wg     = tid % kThreadsPerWg;
    const int wg_warp_id    = tid_in_wg / 32;
    const int lane_in_warp  = tid_in_wg % 32;
    const int frag_row_a    = wg_warp_id * 16 + (lane_in_warp / 4);
    const int frag_col_base = (lane_in_warp % 4) * 2;

    // Consumer wait phase starts at 0 and toggles each ring wrap. At first wait,
    // full[0] is at parity 0 (no producer arrival yet); pass phase=0 to wait
    // for it to flip to 1 once the producer's TMA completes.
    uint32_t phase = 0;
    int      stage = 0;

    for (int tile = blockIdx.x; tile < total_tiles; tile += num_ctas) {
        int tile_m, tile_n;
        tile_to_mn(tile, tile_m, tile_n);
        const int cta_m = tile_m * kBM;
        const int cta_n = tile_n * kBN;

        // Re-init accumulator for each output tile.
        #pragma unroll
        for (int i = 0; i < kAccPerThread; ++i) acc[i] = 0.0f;

        const int num_k_tiles = (K + kBK - 1) / kBK;
        // Deferred-release wgmma pipeline: at iter kt after committing group kt
        // we wait until group kt-(kPipeMmas) has completed.  That stage's smem
        // is no longer being read, so we can release it (mbarrier empty arrive).
        // kPipeMmas = kStages - 1 keeps kStages-1 groups in flight.
        constexpr int kPipeMmas = kStages - 1;
        int release_stage = 0;

        for (int kt = 0; kt < num_k_tiles; ++kt) {
            mbarrier_wait(&full_bar[stage], phase);

            const uint32_t sA_base  = to_smem(sA_stage(stage)  + wg_m_off * kBK);
            const uint32_t sBT_base = to_smem(sBT_stage(stage));

            wgmma_fence();
            #pragma unroll
            for (int ki = 0; ki < kKInner; ++ki) {
                const uint64_t descA = make_smem_desc(sA_base  + ki * 32);
                const uint64_t descB = make_smem_desc(sBT_base + ki * 32);
                wgmma_m64nXk16_bf16<1>(acc, descA, descB);
            }
            wgmma_commit();
            wgmma_wait_group<kPipeMmas>();

            if (kt >= kPipeMmas) {
                if (tid_in_wg == 0) {
                    mbarrier_arrive(&empty_bar[release_stage]);
                }
                release_stage = (release_stage + 1) % kStages;
            }

            stage += 1;
            if (stage == kStages) {
                stage = 0;
                phase ^= 1;
            }
        }
        // Drain any remaining in-flight wgmmas, then release the trailing stages.
        wgmma_wait_group<0>();
        const int remaining = (num_k_tiles < kPipeMmas) ? num_k_tiles : kPipeMmas;
        for (int i = 0; i < remaining; ++i) {
            if (tid_in_wg == 0) {
                mbarrier_arrive(&empty_bar[release_stage]);
            }
            release_stage = (release_stage + 1) % kStages;
        }

        // ---- Epilogue: write 64×BN fp32 fragment to C ------------------------
        // Per CUTLASS CLayout_64xN ((4,8,4),(2,2,N/8)) ((128,1,16),(64,8,512)):
        //   d[4k + 0] = D[row_a][8k + 2c + 0]
        //   d[4k + 1] = D[row_a][8k + 2c + 1]
        //   d[4k + 2] = D[row_b][8k + 2c + 0]
        //   d[4k + 3] = D[row_b][8k + 2c + 1]
        // for k = 0 .. N/8 - 1 (column-pair group index).

        constexpr int kColPairGroups = kBN / 8;     // 16 for BN=128, 24 for BN=192, 32 for BN=256
        const int row_a = cta_m + wg_m_off + frag_row_a;
        const int row_b = row_a + 8;

        if (beta == 0.0f) {
            #pragma unroll
            for (int k = 0; k < kColPairGroups; ++k) {
                const int col0 = cta_n + 8 * k + frag_col_base;
                if (col0 < N) {
                    if (row_a < M) {
                        __nv_bfloat162 v = __floats2bfloat162_rn(
                            alpha * acc[4 * k + 0], alpha * acc[4 * k + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&C[row_a * N + col0]) = v;
                    }
                    if (row_b < M) {
                        __nv_bfloat162 v = __floats2bfloat162_rn(
                            alpha * acc[4 * k + 2], alpha * acc[4 * k + 3]);
                        *reinterpret_cast<__nv_bfloat162*>(&C[row_b * N + col0]) = v;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k = 0; k < kColPairGroups; ++k) {
                const int col0 = cta_n + 8 * k + frag_col_base;
                if (col0 < N) {
                    if (row_a < M) {
                        const float p0 = beta * __bfloat162float(C[row_a * N + col0 + 0]);
                        const float p1 = beta * __bfloat162float(C[row_a * N + col0 + 1]);
                        __nv_bfloat162 v = __floats2bfloat162_rn(
                            alpha * acc[4 * k + 0] + p0, alpha * acc[4 * k + 1] + p1);
                        *reinterpret_cast<__nv_bfloat162*>(&C[row_a * N + col0]) = v;
                    }
                    if (row_b < M) {
                        const float p0 = beta * __bfloat162float(C[row_b * N + col0 + 0]);
                        const float p1 = beta * __bfloat162float(C[row_b * N + col0 + 1]);
                        __nv_bfloat162 v = __floats2bfloat162_rn(
                            alpha * acc[4 * k + 2] + p0,
                            alpha * acc[4 * k + 3] + p1);
                        *reinterpret_cast<__nv_bfloat162*>(&C[row_b * N + col0]) = v;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Host launcher: builds TMA descriptors, sets dynamic SMEM, launches persistent.
// =============================================================================

// The ABI in the build wrapper expects:
//   extern "C" void cuda_claude(const nv_bfloat16*, const nv_bfloat16*, nv_bfloat16*,
//                               int, int, int, float, float)
// We build the TMA descriptors inside this launcher.

namespace {
// Explicit typedef avoids relying on the unversioned PFN_* alias being visible
// in this translation unit (it's gated by some CUDA header macros).
using EncodeTiledFn = CUresult (CUDAAPI *)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*,
    const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// Driver-API entry resolution (works without explicit -lcuda link).
EncodeTiledFn get_tensor_map_encoder() {
    cudaDriverEntryPointQueryResult qr;
    void* fn = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &qr);
    return reinterpret_cast<EncodeTiledFn>(fn);
}

CUtensorMap make_tma_2d_bf16(
    const __nv_bfloat16* gptr,
    uint64_t outer_dim, uint64_t inner_dim_K,
    uint32_t box_outer, uint32_t box_inner_K,
    EncodeTiledFn encode
) {
    CUtensorMap desc{};
    cuuint64_t global_dim[2]   = { inner_dim_K, outer_dim };                  // innermost first
    cuuint64_t global_strides[1] = { inner_dim_K * sizeof(__nv_bfloat16) };   // stride of dim 1
    cuuint32_t box_dim[2]      = { box_inner_K, box_outer };                  // innermost first
    cuuint32_t elem_strides[2] = { 1, 1 };

    encode(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        /*rank=*/2,
        const_cast<__nv_bfloat16*>(gptr),
        global_dim,
        global_strides,
        box_dim,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    return desc;
}
}  // namespace

extern "C" void ENTRYPOINT(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ BT,
    __nv_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    using namespace claude;

    static EncodeTiledFn encode = get_tensor_map_encoder();
    if (encode == nullptr) {
        // Fall back to a noisy abort — caller will see all-zero outputs and the
        // correctness check in the harness will surface this.
        return;
    }

    const CUtensorMap tma_a  = make_tma_2d_bf16(A,  /*outer=M*/ M, /*inner=K*/ K,
                                                /*box_outer=*/ kBM, /*box_inner=*/ kBK, encode);
    const CUtensorMap tma_bt = make_tma_2d_bf16(BT, /*outer=N*/ N, /*inner=K*/ K,
                                                /*box_outer=*/ kBN, /*box_inner=*/ kBK, encode);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    if (sm_count <= 0) sm_count = 132;  // H100 SXM default

    int num_tiles_m = (M + kBM - 1) / kBM;
    int num_tiles_n = (N + kBN - 1) / kBN;
    int total_tiles = num_tiles_m * num_tiles_n;

    int grid_x = sm_count;
    if (grid_x > total_tiles) grid_x = total_tiles;

    dim3 grid(grid_x);
    dim3 block(kNumThreads);

    static bool attr_configured = false;
    if (!attr_configured) {
        cudaFuncSetAttribute(
            (const void*)KERNEL_NAME,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSmemBytes
        );
        attr_configured = true;
    }

    KERNEL_NAME<<<grid, block, kSmemBytes>>>(
        tma_a, tma_bt, C, M, N, K, alpha, beta, num_tiles_m, num_tiles_n
    );
}
