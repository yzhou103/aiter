
// LOCAL STUB: kid901 bpreshuffle experimental header is not committed in this
// branch (never git-added). It is only used by the kid901-913 experimental path,
// which the mxscale flatmm split-K benchmark does not exercise. Disabled to build.
// #include "gfx950/opus_gemm_pipeline_a8w8_blockscale_bpreshuffle_gfx950.cuh"

#if 0  // LOCAL STUB: kid901 bpreshuffle traits need the missing (uncommitted) header
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
#endif  // LOCAL STUB: kid901 bpreshuffle traits


#if 0  // LOCAL STUB: kid901 bpreshuffle device instantiations need the missing header
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
#endif  // LOCAL STUB: kid901 bpreshuffle device instantiations


#if 0  // LOCAL STUB: kid901-913 bpreshuffle impls need the missing (uncommitted) header
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
#endif  // LOCAL STUB: kid901-913 bpreshuffle impls