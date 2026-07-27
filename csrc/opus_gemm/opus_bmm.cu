// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side BMM frontends. These expose BMM/grouped-layout APIs while reusing
// the generated opus GEMM backend launcher symbols.
#include "gfx950/opus_bmm_pipeline_a8w8_mxscale_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh"

// The flatmm-splitk compute-kernel device instantiations (and their traits
// aliases) used to live here. Every tile is now codegen'd into its own
// <tile>_C{void,bf16,fp32}.device.cu (gen_instances_gfx950.py) which owns the
// device symbol, so the manual block here was removed as redundant.

// opus_bmm_splitk_reduce_kernel's definition lives in the shared header
// gfx950/splitk_reduce_gfx950.cuh (pulled in transitively via the flatmm
// split-K pipeline header included at the top of this file) so the codegen'd
// a8w8_mxscale BMM launchers can reference it too.
//
// <VEC=8, BLOCK=128> variant used by the codegen'd split-K launchers, which
// only forward-declare the reduce kernel (fused host TU) and therefore emit a
// stub *reference*. This TU is the single owner of both the device kernel and
// the host __device_stub__ for this specialization, so the instantiations must
// be UNCONDITIONAL (host pass emits the stub definition, device pass the
// kernel). No other TU defines this host stub, so there is no duplicate.
template __global__ void opus_bmm_splitk_reduce_kernel<__bf16, 8, 128>(
    const opus_splitk_ws_handle*, __bf16*,
    int, int, int, int, int, int, int, int);
template __global__ void opus_bmm_splitk_reduce_kernel<float, 8, 128>(
    const opus_splitk_ws_handle*, float*,
    int, int, int, int, int, int, int, int);

#ifndef __HIP_DEVICE_COMPILE__

#include "opus_bmm.h"
#include "opus_gemm_arch.cuh"
#include "opus_build_archs.h"
#include "opus_gemm_manifest.h"
#include "opus_bmm_mxscale_tune_lookup.h"  // GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32
#include "opus_gemm_utils.cuh"  // bf16_t / fp32_t
#include "aiter_stream.h"
#include "gfx950/opus_bmm_launchers_a8w8_mxscale_gfx950.cuh"

#include <unordered_map>

#ifdef OPUS_BUILD_HAS_GFX950
namespace opus_bmm_detail {

// Uniform kid->launcher fn-pointer type. Every kid is codegen'd (no hand-written
// adapters), so this namespace only holds the shared type.
using OpusBmmMxscaleFlatmmSplitkKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, aiter_tensor_t &, int /*splitK*/);
}  // namespace opus_bmm_detail

// Table-driven kid -> launcher dispatch. Launchers come from the generated
// GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32 macro; unknown / untuned kids
// fall back to the 32x128x128 wg2 baseline.
static opus_bmm_detail::OpusBmmMxscaleFlatmmSplitkKernel
opus_bmm_a8w8_mxscale_flatmm_splitk_tune_dispatch(int id)
{
  using namespace opus_bmm_detail;
  static const std::unordered_map<int, OpusBmmMxscaleFlatmmSplitkKernel> kTune = {
      GENERATE_BMM_MXSCALE_FLATMM_SPLITK_LOOKUP_FP32(fp32_t)
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
