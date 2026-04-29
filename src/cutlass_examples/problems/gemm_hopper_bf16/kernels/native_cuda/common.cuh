#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gemm_hopper_bf16::native_cuda {

constexpr int max_bf16_vec_elems(int contiguous_elems) {
    return contiguous_elems % 8 == 0 ? 8 :
           contiguous_elems % 4 == 0 ? 4 :
           contiguous_elems % 2 == 0 ? 2 : 1;
}

template <int VecElems>
struct Bf16Vec;

template <>
struct Bf16Vec<1> {
    using Type = nv_bfloat16;
};

template <>
struct Bf16Vec<2> {
    using Type = unsigned int;
};

template <>
struct Bf16Vec<4> {
    using Type = uint2;
};

template <>
struct Bf16Vec<8> {
    using Type = uint4;
};

template <int VecElems>
__device__ __forceinline__ void copy_bf16_vec(
    nv_bfloat16* __restrict__ dst,
    const nv_bfloat16* __restrict__ src
) {
    using VecType = typename Bf16Vec<VecElems>::Type;
    *reinterpret_cast<VecType*>(dst) = *reinterpret_cast<const VecType*>(src);
}

template <int TileRows, int TileCols, int VecElems, int NumThreads>
__device__ __forceinline__ void load_row_major_tile(
    nv_bfloat16* __restrict__ smem,
    const nv_bfloat16* __restrict__ gmem,
    const int global_row_start,
    const int global_col_start,
    const int global_ld,
    const int valid_rows,
    const int valid_cols,
    const int tid
) {
    static_assert(VecElems == 1 || VecElems == 2 || VecElems == 4 || VecElems == 8);
    static_assert(TileCols % VecElems == 0, "vector copy cannot cross tile rows");

    constexpr int TILE_ELEMENTS = TileRows * TileCols;
    constexpr int THREAD_STRIDE = NumThreads * VecElems;

    for (int linear = tid * VecElems; linear < TILE_ELEMENTS; linear += THREAD_STRIDE) {
        const int row = linear / TileCols;
        const int col = linear % TileCols;

        const int global_offset = (global_row_start + row) * global_ld + global_col_start + col;
        const bool full_vector_in_bounds = row < valid_rows && col + VecElems <= valid_cols;
        const bool global_vector_aligned = global_offset % VecElems == 0;

        if (full_vector_in_bounds && global_vector_aligned) {
            copy_bf16_vec<VecElems>(
                &smem[linear],
                &gmem[global_offset]
            );
        } else {
            #pragma unroll
            for (int i = 0; i < VecElems; ++i) {
                const bool element_in_bounds = row < valid_rows && col + i < valid_cols;
                smem[linear + i] = element_in_bounds
                    ? gmem[global_offset + i]
                    : __float2bfloat16(0.0f);
            }
        }
    }
}

}  // namespace gemm_hopper_bf16::native_cuda
