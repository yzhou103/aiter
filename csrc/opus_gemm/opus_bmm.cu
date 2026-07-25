// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side BMM frontends. These expose BMM/grouped-layout APIs while reusing
// the generated opus GEMM backend launcher symbols.
#include "gfx950/opus_gemm_pipeline_a8w8_scale_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_blockscale_bpreshuffle_gfx950.cuh"

using opus_bmm_a8w8_mxscale_splitk_traits_gfx950 =
    opus_gemm_a8w8_scale_traits_gfx950<512,
      opus::seq<128, 256, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>>;

using opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950 =
    opus_gemm_a8w8_scale_traits_gfx950<512,
      opus::seq<256, 256, 128>,
      opus::tuple<fp8_t, fp8_t, bf16_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>>;

using opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950 =
    opus_gemm_a8w8_scale_traits_gfx950<512,
      opus::seq<256, 256, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>>;

// EXPERIMENTAL (kid 901): independent a8w8 e8m0 blockscale *bpreshuffle*
// (weight direct-to-register). 128x128x128 tile, T_M=2/T_N=2 (256 threads).
// Fully separate from the blockscale (kid150) traits/pipeline above.
// Same tile/wave grid as kid150 (256x256x128, T_M=4, T_N=2, 512 threads) so the
// validated A/mma/epilogue path is reused verbatim; ONLY the B operand path is
// swapped to direct-to-register. Isolates the direct-B effect cleanly.
using opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950 =
    opus_gemm_a8w8_blockscale_bpreshuffle_traits_gfx950<512,
      opus::seq<256, 256, 128>,
      opus::tuple<fp8_t, fp8_t, bf16_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      4, 2>;
using opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950 =
    opus_gemm_a8w8_blockscale_bpreshuffle_traits_gfx950<512,
      opus::seq<256, 256, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      4, 2>;

// kid907: CK-style SMALL tile (128x128x128) on the VALIDATED T_M=4/T_N=2 wave
// grid (512 threads) -- the A-register / mma / epilogue layout helpers were only
// ever derived for T_M=4,T_N=2, so we keep that grid and shrink the tile instead.
// Result: E_M=1 (fast scale path), C accumulator ~32 VGPR, direct-B fragment
// ~32 VGPR -> no VGPR/occupancy wall (vs 256x256's ~128 VGPR C + 64 VGPR B).
using opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950 =
    opus_gemm_a8w8_blockscale_bpreshuffle_traits_gfx950<512,
      opus::seq<128, 128, 128>,
      opus::tuple<fp8_t, fp8_t, bf16_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      4, 2>;
using opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950 =
    opus_gemm_a8w8_blockscale_bpreshuffle_traits_gfx950<512,
      opus::seq<128, 128, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      4, 2>;

// kid913+: CK-class 4-wave 128x128, wave grid 1 M-wave x 4 N-waves (T_M=1/T_N=4).
// This is the config that actually matches CK's compiled blockscale-bpreshuffle
// 128x128: with T_M=1/T_N=4 the half-tile traits give E_M=4/E_N=1 -> 2 halves x
// (4x1) = 8x2 = 16 MFMA/wave = CK's MRepeat=8, NRepeat=2. T_M=1 kills the direct-B
// M-wave redundancy. (The earlier 2x2 attempt had the wrong E-shape and broke.)
// A is read via the mma-derived partition_layout_a (NOT the T_M>=T_N hand-derived
// make_layout_ra, which /0's at T_M=1) in the dedicated gemm_bpre_ck128 kernel.
using opus_bmm_a8w8_bpreshuffle_m128n128k128_ck_bf16_traits_gfx950 =
    opus_gemm_a8w8_blockscale_bpreshuffle_traits_gfx950<256,
      opus::seq<128, 128, 128>,
      opus::tuple<fp8_t, fp8_t, bf16_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1, 4>;
using opus_bmm_a8w8_bpreshuffle_m128n128k128_ck_fp32_traits_gfx950 =
    opus_gemm_a8w8_blockscale_bpreshuffle_traits_gfx950<256,
      opus::seq<128, 128, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1, 4>;

using opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<32, 128, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 128, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm_m64n128k256_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 128, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m128n64k256_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<128, 64, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm256_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<32, 256, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 32, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<32, 64, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 64, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm_m64n64k128_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 64, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<128, 128, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m64n32k256_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 32, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m32n64k256_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<32, 64, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

#ifdef __HIP_DEVICE_COMPILE__
template __global__ void
gemm_a8w8_scale_splitk_kernel<opus_bmm_a8w8_mxscale_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_kernel<opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_kernel<opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// EXPERIMENTAL A-scale preload into LDS (kid157, any K<=8192, K%B_K==0).
template __global__ void
gemm_a8w8_scale_preload_sfa_kernel<
    opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_preload_sfa_kernel<
    opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// EXPERIMENTAL bpreshuffle (kid 901) device instantiations.
template __global__ void
bpre_pack_b_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950>(
    const void*, void*, int, int, int, int, int);
template __global__ void
gemm_bpre_direct_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_direct_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_lds_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_lds_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// kid903: consume standard host-side shuffle_weight(16,16) buffer directly.
template __global__ void
gemm_bpre_shuf_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// kid904: shuffle_weight(16,16) staged through LDS (1x global, conflict-free read).
template __global__ void
gemm_bpre_shuf_lds_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_lds_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// kid905: shuffle_weight(16,16), gfx942-style B reg-prefetch overlapped w/ MMA.
template __global__ void
gemm_bpre_shuf_lds_pf_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_lds_pf_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// kid906: shuffle_weight(16,16), FULL gfx942-style fine-grained pipeline.
template __global__ void
gemm_bpre_shuf_lds_pf2_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_lds_pf2_kernel<opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// kid907: CK-style small-tile (128x128x128) direct-B (shuffle_weight, no LDS).
template __global__ void
gemm_bpre_shuf_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// kid908: small-tile (128x128x128) direct-B + CK-style intrawave reg prefetch.
template __global__ void
gemm_bpre_shuf_direct_pf_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_direct_pf_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// kid909/910/911: small-tile (128x128x128) LDS-B pipelined variants (chase CK's
// BpreShuffle_128x128 = 1470 TFLOPS; kid907 direct-B only reaches 728 because B
// is refetched from global per m-tile with no reuse).
template __global__ void
gemm_bpre_shuf_lds_pf2_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_lds_pf2_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_lds_pf_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_lds_pf_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_lds_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_shuf_lds_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
// kid913: CK-class 4-wave (256-thread, 1 M-wave x 4 N-waves) 128x128 kernel.
template __global__ void
gemm_bpre_ck128_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_ck_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_bpre_ck128_kernel<opus_bmm_a8w8_bpreshuffle_m128n128k128_ck_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_k1024_kernel<opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_k1024_kernel<opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_k1024_lb1_kernel<opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_k1024_lb1_kernel<opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n128k256_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n128k256_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n64k256_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n64k256_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm256_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm256_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm256_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, __bf16, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, float, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, __bf16, 2>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, float, 2>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_minterleave_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_minterleave_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_minterleave_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_minterleave_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true, false, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true, false, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
#endif

// opus_bmm_splitk_reduce_kernel's definition now lives in the shared header
// gfx950/splitk_reduce_gfx950.cuh (pulled in transitively via the flatmm
// split-K pipeline header included at the top of this file) so the codegen'd
// a8w8_mxscale BMM launchers can reference it too. The explicit instantiations
// below stay here: they emit the GPU symbols once for the whole module (both
// the remaining monolithic launchers and the generated split-K launchers link
// against these).
#ifdef __HIP_DEVICE_COMPILE__
template __global__ void opus_bmm_splitk_reduce_kernel<__bf16>(
    const opus_splitk_ws_handle*, __bf16*,
    int, int, int, int, int, int, int, int);
template __global__ void opus_bmm_splitk_reduce_kernel<float>(
    const opus_splitk_ws_handle*, float*,
    int, int, int, int, int, int, int, int);
template __global__ void opus_bmm_splitk_reduce_kernel<__bf16, 8, 128>(
    const opus_splitk_ws_handle*, __bf16*,
    int, int, int, int, int, int, int, int);
template __global__ void opus_bmm_splitk_reduce_kernel<float, 8, 128>(
    const opus_splitk_ws_handle*, float*,
    int, int, int, int, int, int, int, int);
#endif

#ifndef __HIP_DEVICE_COMPILE__

#include "opus_bmm.h"
#include "opus_gemm_arch.cuh"
#include "opus_build_archs.h"
#include "opus_gemm_manifest.h"
#include "opus_bmm_mxscale_tune_lookup.h"  // GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32
#include "opus_gemm_utils.cuh"  // bf16_t / fp32_t
#include "aiter_stream.h"

#include <optional>
#include <unordered_map>

// ── opus_bmm_a8w8_scale() — zero-copy fp8 block-scale BMM ────────────
// O/Y are [M, batch, *] (dim0=M, dim1=batch); x_scale is
// [M, batch, K/GROUP_K] (per-token M). Weight WQ + w_scale stay batch-major
// ([batch, N, K] / [batch, N/GROUP_N, K/GROUP_K]). No caller-side transpose --
// feeds the DSV4 wo_a activation o=[num_tokens, n_groups, K] directly.
#ifdef OPUS_BUILD_HAS_GFX950
template <typename D_C>
void opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template<typename Traits>
__global__ void gemm_a8w8_scale_splitk_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT, bool DIRECT_ONLY>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT, int N_PHASES>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT,
         bool PACK_SCALE_ON_DEMAND>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
#endif

static void opus_bmm_a8w8_common_checks(aiter_tensor_t &O, aiter_tensor_t &wo_a,
                                        aiter_tensor_t &Y, const char *who)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(O.dim() == 3 && wo_a.dim() == 3 && Y.dim() == 3,
              who, ": O/wo_a/Y must be 3D "
              "([M,batch,K] / [batch,N,K] / [M,batch,N])");
  AITER_CHECK(O.dtype() == AITER_DTYPE_fp8 && wo_a.dtype() == AITER_DTYPE_fp8,
              who, ": O and wo_a must be fp8");
  AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32 || Y.dtype() == AITER_DTYPE_bf16,
              who, ": Y must be fp32 or bf16");
  // The kernels index A/B along K with unit stride (kargs carries only M/N/batch
  // strides, never a K stride), so K must be the innermost contiguous dim. The
  // batch axis position is free -- it is fully described by stride_*_batch -- so
  // any batch layout (m-major [M,batch,K], batch-major view, ...) is accepted as
  // long as K stays contiguous. Reject anything else here (host-side, once per
  // launch) rather than silently producing wrong results.
  AITER_CHECK(O.stride(2) == 1, who,
              ": O (x) must be K-contiguous (stride(2)==1); got stride ",
              (long)O.stride(2));
  AITER_CHECK(wo_a.stride(2) == 1, who,
              ": wo_a must be K-contiguous (stride(2)==1); got stride ",
              (long)wo_a.stride(2));
}

void opus_bmm_a8w8_scale(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_scale");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_scale is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<bf16_t>(
        O, wo_a, Y, x_scale, w_scale);
  } else {
    opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<fp32_t>(
        O, wo_a, Y, x_scale, w_scale);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_scale requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_mxscale(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int kernelId)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_mxscale");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_mxscale is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  (void)kernelId;
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<bf16_t>(
        O, wo_a, Y, x_scale, w_scale);
  } else {
    opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<fp32_t>(
        O, wo_a, Y, x_scale, w_scale);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_mxscale_splitk(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_mxscale_splitk");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_mxscale_splitk is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK > 1, "splitK must be > 1");

  using Traits = opus_bmm_a8w8_mxscale_splitk_traits_gfx950;

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  const int split_k = splitK;
  const int total_iters = (K + Traits::B_K - 1) / Traits::B_K;
  const int iters_full = (total_iters + split_k - 1) / split_k;
  const int last_loops = total_iters - (split_k - 1) * iters_full;
  AITER_CHECK(last_loops >= 2,
              "opus_bmm_a8w8_mxscale_splitk requires every split to "
              "have at least 2 K-tiles; K=", K, ", splitK=", split_k,
              ", last split loops=", last_loops);
  const int num_tiles_m = (M + 128 - 1) / 128;
  const int num_tiles_n = (N + 256 - 1) / 256;
  const int padded_M = num_tiles_m * 128;
  const int padded_N = num_tiles_n * 256;

  extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);
  auto stream = aiter::getCurrentHIPStream();
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
  const bool capturing = (capture_status != hipStreamCaptureStatusNone);
  auto* ws_handle = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

  size_t ws_bytes = (size_t)split_k * (size_t)batch
                  * (size_t)padded_M * (size_t)padded_N * sizeof(float);
  if (ws_handle->ptr == nullptr || ws_bytes > ws_handle->bytes) {
    AITER_CHECK(!capturing,
                "splitk workspace grow inside HIP graph capture is not supported");
    void* new_ptr = nullptr;
    const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
    size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
    HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
    if (ws_handle->ptr != nullptr) {
      HIP_CALL(hipDeviceSynchronize());
      HIP_CALL(hipFree(ws_handle->ptr));
    }
    ws_handle->ptr = new_ptr;
    ws_handle->bytes = grow_bytes;
  }

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = ws_handle;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = split_k;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = padded_N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = padded_M * padded_N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
  dim3 block_main(512);
  gemm_a8w8_scale_splitk_kernel<Traits><<<grid_main, block_main, 0, stream>>>(kargs);

  constexpr int REDUCE_VEC = 16;
  constexpr int REDUCE_BS = 64;
  dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                   batch * M, 1);
  dim3 block_reduce(REDUCE_BS);
  const int y_stride_c = (int)Y.stride(0);
  const int y_stride_c_batch = (int)Y.stride(1);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_bmm_splitk_reduce_kernel<__bf16>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<__bf16*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  } else {
    opus_bmm_splitk_reduce_kernel<float>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<float*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale_splitk requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, bool DIRECT_ONLY = false, bool PREFETCH_SCALE = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    bool fusedReduce,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK >= 1, "splitK must be >= 1");
  if constexpr (DIRECT_ONLY) {
    AITER_CHECK(splitK == 1, who, " consumer-self-load kernel requires splitK == 1");
  }
  if (fusedReduce) {
    AITER_CHECK(splitK == 2, who, " fused reduce currently supports splitK == 2 only");
    AITER_CHECK(Traits::B_M == 32 && Traits::B_N == 128 && Traits::B_K == 128,
                who, " fused reduce currently supports the flatmm 32x128x128 variant only");
  }

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  // Scale panels are also read with a unit-stride innermost (K/GROUP_K) dim; the
  // kernel only ever applies stride_sfa/stride_sfb (+ their _batch) offsets, so
  // like A/B the scales must be K-group-contiguous. Batch position stays free.
  AITER_CHECK(x_scale.stride(2) == 1, who,
              ": x_scale must be K-contiguous (stride(2)==1); got stride ",
              (long)x_scale.stride(2));
  AITER_CHECK(w_scale.stride(2) == 1, who,
              ": w_scale must be K-contiguous (stride(2)==1); got stride ",
              (long)w_scale.stride(2));
  // Partial M tiles are handled in-kernel via buffer OOB masking (A/sfa reads
  // return 0 and C stores are dropped for rows >= M), so small tiles need not
  // have M % B_M == 0. Large tiles (B_M >= 128) would waste most of a 128/256
  // row tile on a tiny M and are never the right pick there, so we keep the
  // divisibility requirement for them (callers/tuner select a B_M <= 64 tile
  // for small M instead). N and K must still tile exactly (B/K axes unbounded).
  if constexpr (Traits::B_M >= 128) {
    AITER_CHECK(M % Traits::B_M == 0,
                "flatmm splitK v1 (B_M>=128 tile) requires M % ", Traits::B_M,
                " == 0, got ", M, " (use a B_M<=64 tile for small M)");
  }
  AITER_CHECK(N % Traits::B_N == 0,
              "flatmm splitK v1 requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              "flatmm splitK v1 requires K % ", Traits::B_K, " == 0, got ", K);

  const int split_k = splitK;
  // Match the rest of opus GEMM: splitK is the number of K partitions.
  // splitK == 1 means no K split, so the main kernel writes Y directly.
  const bool no_split_k = (split_k == 1);
  const int total_iters = K / Traits::B_K;
  const int iters_full = (total_iters + split_k - 1) / split_k;
  const int last_loops = total_iters - (split_k - 1) * iters_full;
  AITER_CHECK(last_loops >= Traits::prefetch_k_iter,
              "flatmm splitK v1 requires every split to have at least ",
              Traits::prefetch_k_iter, " K-tiles; K=", K,
              " gives total_iters=", total_iters, ", splitK=", split_k,
              ", last split loops=", last_loops);

  const int num_tiles_m = (M + Traits::B_M - 1) / Traits::B_M;
  const int num_tiles_n = (N + Traits::B_N - 1) / Traits::B_N;
  const int padded_M = num_tiles_m * Traits::B_M;
  const int padded_N = num_tiles_n * Traits::B_N;
  const size_t partial_bytes = (size_t)split_k * (size_t)batch
                             * (size_t)padded_M * (size_t)padded_N * sizeof(float);
  const size_t counter_offset = (partial_bytes + 255) & ~((size_t)255);
  const size_t counter_bytes = (size_t)batch * (size_t)num_tiles_m
                             * (size_t)num_tiles_n * sizeof(int);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = split_k;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = padded_N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = padded_M * padded_N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (no_split_k) {
    kargs.ptr_c = Y.data_ptr();
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    if (Y.dtype() == AITER_DTYPE_bf16) {
      gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, __bf16, DIRECT_ONLY, PREFETCH_SCALE>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, float, DIRECT_ONLY, PREFETCH_SCALE>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    }
    return;
  }

  extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
  const bool capturing = (capture_status != hipStreamCaptureStatusNone);
  auto* ws_handle = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

  const size_t ws_bytes = fusedReduce ? (counter_offset + counter_bytes) : partial_bytes;
  if (ws_handle->ptr == nullptr || ws_bytes > ws_handle->bytes) {
    AITER_CHECK(!capturing,
                "splitk workspace grow inside HIP graph capture is not supported");
    void* new_ptr = nullptr;
    const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
    size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
    HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
    if (ws_handle->ptr != nullptr) {
      HIP_CALL(hipDeviceSynchronize());
      HIP_CALL(hipFree(ws_handle->ptr));
    }
    ws_handle->ptr = new_ptr;
    ws_handle->bytes = grow_bytes;
  }
  kargs.ws_handle = ws_handle;

  if (fusedReduce) {
    kargs.ptr_c = Y.data_ptr();
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    kargs.counter_offset_bytes = counter_offset;
    HIP_CALL(hipMemsetAsync(static_cast<char*>(ws_handle->ptr) + counter_offset,
                            0, counter_bytes, stream));
    if (Y.dtype() == AITER_DTYPE_bf16) {
      gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, __bf16>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, float>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    }
    return;
  }

  gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits>
      <<<grid_main, block_main, 0, stream>>>(kargs);

  constexpr int REDUCE_VEC = 8;
  constexpr int REDUCE_BS = 128;
  dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                   batch * M, 1);
  dim3 block_reduce(REDUCE_BS);
  const int y_stride_c = (int)Y.stride(0);
  const int y_stride_c_batch = (int)Y.stride(1);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_bmm_splitk_reduce_kernel<__bf16, REDUCE_VEC, REDUCE_BS>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<__bf16*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  } else {
    opus_bmm_splitk_reduce_kernel<float, REDUCE_VEC, REDUCE_BS>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<float*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, int N_PHASES>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_nphase_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_N = Traits::B_N * N_PHASES;
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % LOGICAL_B_N == 0,
              who, " requires N % ", LOGICAL_B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);
  const int total_iters = K / Traits::B_K;
  AITER_CHECK(total_iters >= Traits::prefetch_k_iter,
              who, " requires at least ", Traits::prefetch_k_iter,
              " K-tiles, got ", total_iters);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = 1;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / LOGICAL_B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel<Traits, __bf16, N_PHASES>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel<Traits, float, N_PHASES>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, bool SKIP_SCALE_WAIT = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              who, " requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);
  const int total_iters = K / Traits::B_K;
  AITER_CHECK(total_iters >= Traits::prefetch_k_iter,
              who, " requires at least ", Traits::prefetch_k_iter,
              " K-tiles, got ", total_iters);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / Traits::B_N;
  const int m_per_wg = (num_tiles_m >= 16) ? 2 : 1;
  kargs.split_k = m_per_wg;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int split_m = (num_tiles_m + m_per_wg - 1) / m_per_wg;
  constexpr int NUM_XCD = 8;
  const int m_grp_per_xcd = (split_m + NUM_XCD - 1) / NUM_XCD;
  kargs.stride_ws = split_m;
  kargs.stride_ws_batch = m_grp_per_xcd;
  dim3 grid_main(NUM_XCD * m_grp_per_xcd * num_tiles_n, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<Traits, __bf16, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<Traits, float, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

// Tunable variant of the persistent mouter kernel: the API `splitK` argument is
// repurposed to drive m_per_wg directly (>=1), so we can sweep how many M-tiles
// each workgroup streams back-to-back. Reuses the same mouter kernel template as
// kid 131/144, so no additional explicit instantiations are required.
template <typename Traits, bool SKIP_SCALE_WAIT = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_tunable_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK >= 1, who, " requires splitK (=m_per_wg) >= 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              who, " requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);
  const int total_iters = K / Traits::B_K;
  AITER_CHECK(total_iters >= Traits::prefetch_k_iter,
              who, " requires at least ", Traits::prefetch_k_iter,
              " K-tiles, got ", total_iters);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / Traits::B_N;
  int m_per_wg = splitK;
  if (m_per_wg > num_tiles_m) m_per_wg = num_tiles_m;
  if (m_per_wg < 1) m_per_wg = 1;
  kargs.split_k = m_per_wg;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int split_m = (num_tiles_m + m_per_wg - 1) / m_per_wg;
  constexpr int NUM_XCD = 8;
  const int m_grp_per_xcd = (split_m + NUM_XCD - 1) / NUM_XCD;
  kargs.stride_ws = split_m;
  kargs.stride_ws_batch = m_grp_per_xcd;
  dim3 grid_main(NUM_XCD * m_grp_per_xcd * num_tiles_n, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<Traits, __bf16, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<Traits, float, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

// M-tile interleaved kernel host wrapper. MI=2 consecutive M tiles per WG share
// the B stream; requires M % (MI * B_M) == 0. splitK arg is unused (must be 1).
template <typename Traits, bool SKIP_SCALE_WAIT = false>
static void opus_bmm_a8w8_mxscale_flatmm_minterleave_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  constexpr int MI = 2;
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % (MI * Traits::B_M) == 0,
              who, " requires M % ", (MI * Traits::B_M), " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              who, " requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);
  const int total_iters = K / Traits::B_K;
  AITER_CHECK(total_iters >= Traits::prefetch_k_iter,
              who, " requires at least ", Traits::prefetch_k_iter,
              " K-tiles, got ", total_iters);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / Traits::B_N;
  kargs.split_k = MI;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int split_m = num_tiles_m / MI;          // M-tile groups (WGs along M)
  constexpr int NUM_XCD = 8;
  const int m_grp_per_xcd = (split_m + NUM_XCD - 1) / NUM_XCD;
  kargs.stride_ws = split_m;
  kargs.stride_ws_batch = m_grp_per_xcd;
  dim3 grid_main(NUM_XCD * m_grp_per_xcd * num_tiles_n, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_minterleave_kernel<Traits, __bf16, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_minterleave_kernel<Traits, float, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_wave8n2_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_N = Traits::B_N * 2;
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % LOGICAL_B_N == 0,
              who, " requires N % ", LOGICAL_B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = 1;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / LOGICAL_B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(512);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel<Traits, __bf16>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel<Traits, float>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, bool ISSUE_NEXT_BEFORE_SCALE = false,
          bool SKIP_SCALE_WAIT = false, bool SINGLE_LDS_SLOT = false,
          bool ISSUE_NEXT_AFTER_MMA = false, bool PACK_SCALE_ON_DEMAND = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_N = Traits::B_N * 2;
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % LOGICAL_B_N == 0,
              who, " requires N % ", LOGICAL_B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = 1;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / LOGICAL_B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(256);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
        Traits, __bf16, ISSUE_NEXT_BEFORE_SCALE, SKIP_SCALE_WAIT, SINGLE_LDS_SLOT,
        ISSUE_NEXT_AFTER_MMA, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
        Traits, float, ISSUE_NEXT_BEFORE_SCALE, SKIP_SCALE_WAIT, SINGLE_LDS_SLOT,
        ISSUE_NEXT_AFTER_MMA, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, bool SKIP_SCALE_WAIT = false,
          bool PACK_SCALE_ON_DEMAND = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_M = Traits::B_M * 2;
  AITER_CHECK(M % LOGICAL_B_M == 0,
              who, " requires M % ", LOGICAL_B_M, " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              who, " requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = 1;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int num_tiles_m = M / LOGICAL_B_M;
  const int num_tiles_n = N / Traits::B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(256);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
        Traits, __bf16, SKIP_SCALE_WAIT, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
        Traits, float, SKIP_SCALE_WAIT, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Bf16Traits, typename Fp32Traits, bool K1024_ONLY = false,
          bool K1024_LB1 = false, bool PRELOAD_SFA_LDS = false>
static void opus_bmm_a8w8_mxscale_pipeline_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Bf16Traits::B_M == 0,
              who, " requires M % ", Bf16Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % Bf16Traits::B_N == 0,
              who, " requires N % ", Bf16Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Bf16Traits::B_K == 0,
              who, " requires K % ", Bf16Traits::B_K, " == 0, got ", K);
  if constexpr (K1024_ONLY || K1024_LB1) {
    AITER_CHECK(K == 1024, who, " requires K == 1024, got ", K);
  }

  opus_gemm_scale_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ptr_c = Y.data_ptr();
  kargs.m = M;
  kargs.n = N;
  kargs.k = K;
  kargs.batch = batch;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  const int num_tiles_m = M / Bf16Traits::B_M;
  const int num_tiles_n = N / Bf16Traits::B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(Bf16Traits::BLOCK_SIZE);
  auto stream = aiter::getCurrentHIPStream();
  if (Y.dtype() == AITER_DTYPE_bf16) {
    if constexpr (PRELOAD_SFA_LDS) {
      gemm_a8w8_scale_preload_sfa_kernel<Bf16Traits>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    } else if constexpr (K1024_LB1) {
      gemm_a8w8_scale_k1024_lb1_kernel<Bf16Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    } else if constexpr (K1024_ONLY) {
      gemm_a8w8_scale_k1024_kernel<Bf16Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      gemm_a8w8_scale_kernel<Bf16Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    }
  } else {
    if constexpr (PRELOAD_SFA_LDS) {
      gemm_a8w8_scale_preload_sfa_kernel<Fp32Traits>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    } else if constexpr (K1024_LB1) {
      gemm_a8w8_scale_k1024_lb1_kernel<Fp32Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    } else if constexpr (K1024_ONLY) {
      gemm_a8w8_scale_k1024_kernel<Fp32Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      gemm_a8w8_scale_kernel<Fp32Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    }
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

#ifdef OPUS_BUILD_HAS_GFX950
namespace opus_bmm_detail {

// Uniform launcher signature shared by the codegen'd flatmm-splitk launchers
// (opus_gemm_manifest.h) and every specialized pipeline wrapper below. This is
// the BMM analogue of opus_gfx942_detail::OpusA8W8BlockscaleBPreshuffleKernel:
// a plain function pointer so kid->kernel dispatch is a single table lookup
// (see opus_bmm_a8w8_mxscale_flatmm_splitk_tune_dispatch), mirroring
// opus_gemm.cu's opus_a8w8_tune_dispatch_gfx942.
using OpusBmmMxscaleFlatmmSplitkKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, aiter_tensor_t &, int /*splitK*/);

// ── Specialized (non-codegen) pipeline wrappers ────────────────────────────
// The big-tile mouter / minterleave / wave*n* / nphase / scale-pipeline / fused
// families stay hand-written (extra compile-time flags + a diagnostic `who`
// string), so they can't be emitted by gen_instances. Each wrapper adapts one
// _impl instantiation to the uniform launcher signature above, letting it sit
// in the same kid->fn table as the codegen'd launchers.
using _bmm_flatmm_wg1_traits =
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950;

static void bmm_kid_129(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_nphase_mmajor_impl<
      opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, 2>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m64n256k128_nphase)");
}
static void bmm_kid_131(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_mmajor_impl<_bmm_flatmm_wg1_traits>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n128k128_persistent_mouter_wg1)");
}
static void bmm_kid_144(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_mmajor_impl<_bmm_flatmm_wg1_traits, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n128k128_persistent_mouter_wg1_skip_scale_wait)");
}
static void bmm_kid_160(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_tunable_mmajor_impl<_bmm_flatmm_wg1_traits>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n128k128_persistent_mouter_wg1_tunable)");
}
static void bmm_kid_161(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_tunable_mmajor_impl<_bmm_flatmm_wg1_traits, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n128k128_persistent_mouter_wg1_tunable_skip_scale_wait)");
}
static void bmm_kid_162(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_minterleave_mmajor_impl<_bmm_flatmm_wg1_traits>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n128k128_minterleave_mi2)");
}
static void bmm_kid_163(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_minterleave_mmajor_impl<_bmm_flatmm_wg1_traits, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n128k128_minterleave_mi2_skip_scale_wait)");
}
static void bmm_kid_132(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave8n2_mmajor_impl<_bmm_flatmm_wg1_traits>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n256k128_wave8n2)");
}
static void bmm_kid_133(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n256k128_wave4n2_selfload)");
}
static void bmm_kid_140(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n256k128_wave4n2_selfload_issue_next)");
}
static void bmm_kid_141(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits, false, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n256k128_wave4n2_selfload_skip_scale_wait)");
}
static void bmm_kid_145(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits, false, true, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n256k128_wave4n2_selfload_single_lds)");
}
static void bmm_kid_146(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits, false, true, false, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n256k128_wave4n2_selfload_issue_after_mma)");
}
static void bmm_kid_147(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits, false, true, false, false, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m128n256k128_wave4n2_selfload_on_demand_scale_pack)");
}
static void bmm_kid_134(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m256n128k128_wave4m2_selfload)");
}
static void bmm_kid_142(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m256n128k128_wave4m2_selfload_skip_scale_wait)");
}
static void bmm_kid_148(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_mmajor_impl<_bmm_flatmm_wg1_traits, true, true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m256n128k128_wave4m2_selfload_on_demand_scale_pack)");
}
static void bmm_kid_149(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  AITER_CHECK(splitK == 1,
              "opus_bmm_a8w8_mxscale_flatmm_splitk(m512n256k256_scale_pipeline) requires splitK == 1");
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<bf16_t>(
        O, wo_a, Y, x_scale, w_scale);
  } else {
    opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<fp32_t>(
        O, wo_a, Y, x_scale, w_scale);
  }
}
static void bmm_kid_150(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_pipeline_mmajor_impl<
      opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950,
      opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m256n256k128_scale_pipeline)");
}
// kid157: kid150 with the whole A-scale panel preloaded into LDS once, reading
// per-tile A-scale from LDS in the main loop (any K<=8192, K%B_K==0).
static void bmm_kid_157(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_pipeline_mmajor_impl<
      opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950,
      opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950,
      false,
      false,
      true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m256n256k128_preload_sfa_lds)");
}
static void bmm_kid_151(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_pipeline_mmajor_impl<
      opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950,
      opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950,
      true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m256n256k1024_scale_pipeline)");
}
static void bmm_kid_152(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_pipeline_mmajor_impl<
      opus_bmm_a8w8_mxscale_m256n256k128_bf16_traits_gfx950,
      opus_bmm_a8w8_mxscale_m256n256k128_fp32_traits_gfx950,
      false,
      true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(m256n256k1024_scale_pipeline_lb1)");
}
// kid 100: the ONLY fused-reduce path (splitK==2 counter variant, 32x128x128
// wg2). Traits are identical to kid 0/32, so its main-kernel symbols resolve to
// the generated device TUs; only the fused host launcher stays hand-written.
static void bmm_kid_100(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_mxscale_flatmm_splitk_impl<
      opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      true,
      "opus_bmm_a8w8_mxscale_flatmm_splitk(fused)");
}

// EXPERIMENTAL kid 901: a8w8 e8m0 blockscale bpreshuffle (weight direct-to-
// register). Independent pipeline. Packs W -> B_pre (cached scratch), then runs
// the direct-B compute kernel. NOTE: correctness-first -- packing runs every
// call (it will show up in device-time perf; a pack-once path comes next).
template<typename Bf16Traits, typename Fp32Traits, bool USE_LDS>
static void opus_bmm_a8w8_bpreshuffle_direct_impl(
    aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
    aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK, const char* who) {
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950, who, " is gfx950-only");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Bf16Traits::B_M == 0, who, " requires M % ", Bf16Traits::B_M, " == 0");
  AITER_CHECK(N % Bf16Traits::B_N == 0, who, " requires N % ", Bf16Traits::B_N, " == 0");
  AITER_CHECK(K % Bf16Traits::B_K == 0, who, " requires K % ", Bf16Traits::B_K, " == 0");

  const int num_tiles_m = M / Bf16Traits::B_M;
  const int num_tiles_n = N / Bf16Traits::B_N;
  const int loops = K / Bf16Traits::B_K;
  auto stream = aiter::getCurrentHIPStream();

  // Cached B_pre scratch (per-process, single stream; experiment only).
  const size_t bpre_bytes = (size_t)batch * num_tiles_n * loops
                          * Bf16Traits::b_direct_vecs_per_ktile * Bf16Traits::VEC_B;
  static void* s_bpre_ptr = nullptr;
  static size_t s_bpre_bytes = 0;
  if (s_bpre_ptr == nullptr || bpre_bytes > s_bpre_bytes) {
    if (s_bpre_ptr) { HIP_CALL(hipDeviceSynchronize()); HIP_CALL(hipFree(s_bpre_ptr)); }
    HIP_CALL(hipMalloc(&s_bpre_ptr, bpre_bytes));
    s_bpre_bytes = bpre_bytes;
  }

  const int sb = (int)wo_a.stride(1);
  const int sb_batch = (int)wo_a.stride(0);
  dim3 grid_pack(num_tiles_n, 1, batch);
  dim3 block(Bf16Traits::BLOCK_SIZE);
  // Pack-once: the weight preshuffle is a one-time transform. Skip re-packing
  // when the same weight buffer/shape is seen again (fair perf: packing lands
  // in warmup, the timed loop is pure compute). Re-pack if W ptr/shape change.
  static const void* s_last_wsrc = nullptr;
  static int s_last_n = -1, s_last_k = -1, s_last_batch = -1;
  const bool need_pack = (wo_a.data_ptr() != s_last_wsrc) || N != s_last_n
                       || K != s_last_k || batch != s_last_batch;
#ifndef BPRE_LDS_B
  if (need_pack) {
    bpre_pack_b_kernel<Bf16Traits><<<grid_pack, block, 0, stream>>>(
        wo_a.data_ptr(), s_bpre_ptr, N, K, batch, sb, sb_batch);
    s_last_wsrc = wo_a.data_ptr(); s_last_n = N; s_last_k = K; s_last_batch = batch;
  }
#endif

  opus_gemm_scale_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
#ifdef BPRE_LDS_B
  kargs.ptr_b = wo_a.data_ptr();  // DEBUG: compute reads original W via LDS path
#else
  kargs.ptr_b = s_bpre_ptr;
#endif
  kargs.ptr_c = Y.data_ptr();
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = sb;
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = sb_batch;
  kargs.stride_c_batch = (int)Y.stride(1);
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    if constexpr (USE_LDS) gemm_bpre_lds_kernel<Bf16Traits><<<grid_main, block, 0, stream>>>(kargs);
    else                   gemm_bpre_direct_kernel<Bf16Traits><<<grid_main, block, 0, stream>>>(kargs);
  } else {
    if constexpr (USE_LDS) gemm_bpre_lds_kernel<Fp32Traits><<<grid_main, block, 0, stream>>>(kargs);
    else                   gemm_bpre_direct_kernel<Fp32Traits><<<grid_main, block, 0, stream>>>(kargs);
  }
}

static void bmm_kid_901(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_bpreshuffle_direct_impl<
      opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950,
      opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950, /*USE_LDS=*/false>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m256n256k128_direct_b)");
}

// kid 902: same pre-shuffled B, but staged through LDS (1x global traffic like
// kid150, but contiguous no-swizzle load/read). vs kid150 = "does bpreshuffle
// simplify the LDS B path enough to win?".
static void bmm_kid_902(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  opus_bmm_a8w8_bpreshuffle_direct_impl<
      opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950,
      opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950, /*USE_LDS=*/true>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m256n256k128_lds_b)");
}

// kid 903/904: B consumed from the STANDARD host-side shuffle_weight(w,(16,16))
// buffer (python shuffles wo_a before the call). No device packer -- ptr_b IS
// the shuffled weight. USE_LDS=false (903) reads B directly from global
// (validates the gfx950 MFMA B-fragment <-> shuffle_weight mapping; has T_M
// redundant global traffic). USE_LDS=true (904) stages B through LDS -> 1x
// global traffic + conflict-free, no-swizzle read (the perf candidate vs kid150).
// MODE: 0 = direct (kid903), 1 = LDS (kid904), 2 = LDS + gfx942-style reg-prefetch (kid905)
// The tile traits are template params so a single body serves both the 256x256
// variants (kid903-906) and the CK-style small 128x128 tile (kid907).
template<int MODE,
         typename Bf16Traits = opus_bmm_a8w8_bpreshuffle_m256n256k128_bf16_traits_gfx950,
         typename Fp32Traits = opus_bmm_a8w8_bpreshuffle_m256n256k128_fp32_traits_gfx950>
static void bmm_kid_shuf_impl(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK,
                        const char* who) {
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950, who, " is gfx950-only");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Bf16Traits::B_M == 0, who, " requires M % ", Bf16Traits::B_M, " == 0");
  AITER_CHECK(N % Bf16Traits::B_N == 0, who, " requires N % ", Bf16Traits::B_N, " == 0");
  AITER_CHECK(K % Bf16Traits::B_K == 0, who, " requires K % ", Bf16Traits::B_K, " == 0");

  const int num_tiles_m = M / Bf16Traits::B_M;
  const int num_tiles_n = N / Bf16Traits::B_N;
  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();  // already shuffle_weight(16,16)'d on host
  kargs.ptr_c = Y.data_ptr();
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block(Bf16Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    if      constexpr (MODE == 5) gemm_bpre_ck128_kernel<Bf16Traits><<<grid_main, block, 0, stream>>>(kargs);
    else if constexpr (MODE == 4) gemm_bpre_shuf_direct_pf_kernel<Bf16Traits><<<grid_main, block, 0, stream>>>(kargs);
    else if constexpr (MODE == 3) gemm_bpre_shuf_lds_pf2_kernel<Bf16Traits><<<grid_main, block, 0, stream>>>(kargs);
    else if constexpr (MODE == 2) gemm_bpre_shuf_lds_pf_kernel<Bf16Traits><<<grid_main, block, 0, stream>>>(kargs);
    else if constexpr (MODE == 1) gemm_bpre_shuf_lds_kernel<Bf16Traits><<<grid_main, block, 0, stream>>>(kargs);
    else                          gemm_bpre_shuf_kernel<Bf16Traits><<<grid_main, block, 0, stream>>>(kargs);
  } else {
    if      constexpr (MODE == 5) gemm_bpre_ck128_kernel<Fp32Traits><<<grid_main, block, 0, stream>>>(kargs);
    else if constexpr (MODE == 4) gemm_bpre_shuf_direct_pf_kernel<Fp32Traits><<<grid_main, block, 0, stream>>>(kargs);
    else if constexpr (MODE == 3) gemm_bpre_shuf_lds_pf2_kernel<Fp32Traits><<<grid_main, block, 0, stream>>>(kargs);
    else if constexpr (MODE == 2) gemm_bpre_shuf_lds_pf_kernel<Fp32Traits><<<grid_main, block, 0, stream>>>(kargs);
    else if constexpr (MODE == 1) gemm_bpre_shuf_lds_kernel<Fp32Traits><<<grid_main, block, 0, stream>>>(kargs);
    else                          gemm_bpre_shuf_kernel<Fp32Traits><<<grid_main, block, 0, stream>>>(kargs);
  }
}

static void bmm_kid_903(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/0>(O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m256n256k128_shuffle_weight_direct)");
}

static void bmm_kid_904(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/1>(O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m256n256k128_shuffle_weight_lds)");
}

static void bmm_kid_905(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/2>(O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m256n256k128_shuffle_weight_lds_prefetch)");
}

static void bmm_kid_906(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/3>(O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m256n256k128_shuffle_weight_lds_pipeline)");
}

// kid907: CK-style small tile (128x128x128) direct-B (shuffle_weight, no LDS).
static void bmm_kid_907(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/0,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m128n128k128_shuffle_weight_direct)");
}

// kid908: small-tile (128x128x128) direct-B + intrawave register prefetch.
static void bmm_kid_908(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/4,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m128n128k128_shuffle_weight_direct_prefetch)");
}

// kid909: small-tile (128x128x128) LDS-B FULL gfx942-style fine-grained pipeline
// (MODE=3). This is the 128x128 analogue of kid906 -- the most CK-like variant.
static void bmm_kid_909(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/3,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m128n128k128_shuffle_weight_lds_pipeline)");
}

// kid910: small-tile (128x128x128) LDS-B + gfx942-style reg-prefetch (MODE=2).
static void bmm_kid_910(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/2,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m128n128k128_shuffle_weight_lds_prefetch)");
}

// kid911: small-tile (128x128x128) LDS-B staged, no prefetch (MODE=1).
static void bmm_kid_911(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/1,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_bf16_traits_gfx950,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_fp32_traits_gfx950>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m128n128k128_shuffle_weight_lds)");
}

// kid913: CK-class 4-wave 128x128, 1 M-wave x 4 N-waves (T_M=1/T_N=4), MODE=5
// (gemm_bpre_ck128_kernel: A via mma partition_layout_a, B via read_b_lds). Chases CK 1470.
static void bmm_kid_913(aiter_tensor_t &O, aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                        aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, int splitK) {
  bmm_kid_shuf_impl</*MODE=*/5,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_ck_bf16_traits_gfx950,
      opus_bmm_a8w8_bpreshuffle_m128n128k128_ck_fp32_traits_gfx950>(
      O, wo_a, Y, x_scale, w_scale, splitK,
      "opus_bmm_a8w8_blockscale_bpreshuffle(m128n128k128_ck_1x4wave)");
}

}  // namespace opus_bmm_detail

// Table-driven kid -> launcher dispatch, mirroring opus_gemm.cu's
// opus_a8w8_tune_dispatch_gfx942. The codegen'd launchers come from the
// generated GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32 macro (kid numbers
// preserved from the historical switch); the specialized pipelines are appended
// via the wrappers above. Unknown / untuned kids fall back to the
// guaranteed-runnable 32x128x128 wg2 baseline (the old switch's `default`).
static opus_bmm_detail::OpusBmmMxscaleFlatmmSplitkKernel
opus_bmm_a8w8_mxscale_flatmm_splitk_tune_dispatch(int id)
{
  using namespace opus_bmm_detail;
  static const std::unordered_map<int, OpusBmmMxscaleFlatmmSplitkKernel> kTune = {
      GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32(fp32_t)
      {129, &bmm_kid_129}, {131, &bmm_kid_131}, {144, &bmm_kid_144},
      {160, &bmm_kid_160}, {161, &bmm_kid_161}, {162, &bmm_kid_162},
      {163, &bmm_kid_163}, {132, &bmm_kid_132}, {133, &bmm_kid_133},
      {140, &bmm_kid_140}, {141, &bmm_kid_141}, {145, &bmm_kid_145},
      {146, &bmm_kid_146}, {147, &bmm_kid_147}, {134, &bmm_kid_134},
      {142, &bmm_kid_142}, {148, &bmm_kid_148}, {149, &bmm_kid_149},
      {150, &bmm_kid_150}, {151, &bmm_kid_151}, {152, &bmm_kid_152},
      {157, &bmm_kid_157},
      {100, &bmm_kid_100}, {901, &bmm_kid_901}, {902, &bmm_kid_902},
      {903, &bmm_kid_903}, {904, &bmm_kid_904}, {905, &bmm_kid_905},
      {906, &bmm_kid_906}, {907, &bmm_kid_907}, {908, &bmm_kid_908},
      {909, &bmm_kid_909}, {910, &bmm_kid_910}, {911, &bmm_kid_911},
      {913, &bmm_kid_913},
  };
  auto it = kTune.find(id);
  if (it != kTune.end())
    return it->second;
  return &opus_bmm_a8w8_mxscale_flatmm_splitk_256x32x128x128_2x1_16x16x128_1x128x128_wgpcu2<fp32_t>;
}
#endif  // OPUS_BUILD_HAS_GFX950

void opus_bmm_a8w8_mxscale_flatmm_splitk(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    int kernelId)
{
  // Common dtype/shape validation + arch gate, done once here so the codegen'd
  // launchers (which omit these to stay lean) and the fused kid 100 wrapper share
  // one check. The _impl still re-checks internally (idempotent).
  opus_bmm_a8w8_common_checks(O, wo_a, Y,
                              "opus_bmm_a8w8_mxscale_flatmm_splitk");
#ifndef OPUS_BUILD_HAS_GFX950
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale_flatmm_splitk requires "
              "OPUS_BUILD_HAS_GFX950");
#else
  {
    const auto &arch_info = opus_get_arch_info();
    AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
                "opus_bmm_a8w8_mxscale_flatmm_splitk is gfx950-only; "
                "current device ", arch_info.dev, " has gcnArchName='",
                arch_info.name, "'");
  }
  // Single table lookup instead of a ~40-case switch (see opus_gemm.cu).
  opus_bmm_a8w8_mxscale_flatmm_splitk_tune_dispatch(kernelId)(
      O, wo_a, Y, x_scale, w_scale, splitK);
#endif  // OPUS_BUILD_HAS_GFX950
}

#endif  // !__HIP_DEVICE_COMPILE__
