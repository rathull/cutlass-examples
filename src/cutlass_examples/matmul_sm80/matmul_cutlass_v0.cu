// CUTLASS 2.x-style device-level GEMM for SM80 (Ampere).
// Uses cutlass::gemm::device::Gemm with bf16 tensor core MMA and fp32 accumulation.
// The 3.x CollectiveBuilder + kernel::GemmUniversal API requires SM90+ (Hopper).

#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using Element = cutlass::bfloat16_t;
using ElementAccumulator = float;

// A is M×K row-major, B is N×K row-major (≡ K×N col-major), C is M×N row-major.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<Element>::value;  // 8
static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<Element>::value;  // 8

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element,
    kAlignmentA,
    ElementAccumulator,
    ElementAccumulator
>;

using Gemm = cutlass::gemm::device::Gemm<
    Element, LayoutA,
    Element, LayoutB,
    Element, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,   // CTA tile
    cutlass::gemm::GemmShape<64, 64, 32>,     // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,      // MMA op (bf16 m16n8k16)
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,          // pipeline stages
    kAlignmentA,
    kAlignmentB
>;

void matmul_cutlass_v0(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    auto a = reinterpret_cast<const Element*>(A);
    auto b = reinterpret_cast<const Element*>(B);
    auto c = reinterpret_cast<Element*>(C);

    typename Gemm::Arguments args{
        {M, N, K},
        {a, K},         // A: M×K row-major, ld = K
        {b, K},         // B: K×N col-major, ld = K
        {c, N},         // C (source): M×N row-major, ld = N
        {c, N},         // D (destination): same as C
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
        1               // split-k slices
    };

    Gemm gemm;
    gemm(args);
}
