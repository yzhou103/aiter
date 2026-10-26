// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "kernel.h"
#include <cstdint>

// Semantic eligibility for prefill specializations. T/capacity performance
// thresholds live separately in dispatch_policy.hpp.
namespace fused_norm_rope_attn_rope_cast_fwd {

constexpr int packed_data_bytes  = 512;
constexpr int packed_scale_bytes = 16;
constexpr int packed_row_bytes   = packed_data_bytes + packed_scale_bytes;

inline bool kv_offsets_fit_u32(const FusedNormRopeAttnRopeCastParams& a)
{
    // Guard both sources and their independent scale views; bases remain 64-bit.
    return int64_t(a.total_pages) * a.stride_kv_page <= 0xffffffffLL &&
           int64_t(a.total_pages) * a.stride_kv_scale_page <= 0xffffffffLL &&
           int64_t(a.total_tokens) * a.stride_kv_extend <= 0xffffffffLL &&
           int64_t(a.total_tokens) * a.stride_kv_scale_extend <= 0xffffffffLL;
}

inline bool supports_packed_prefill(const FusedNormRopeAttnRopeCastParams& a, bool raw_q)
{
    return !a.q_norm && !a.positions64 && a.q_is_rotated == !raw_q && !a.out_ptr && !a.debug_ptr &&
           !a.max_ptr && a.attn_sink_ptr && a.lse_ptr && a.stride_kv_page == packed_row_bytes &&
           a.stride_kv_scale_page == packed_row_bytes && a.stride_kv_extend == packed_row_bytes &&
           a.stride_kv_scale_extend == packed_row_bytes &&
           a.unified_kv_scale_ptr ==
               static_cast<const char*>(a.unified_kv_ptr) + packed_data_bytes &&
           a.kv_scale_ptr == static_cast<const char*>(a.kv_ptr) + packed_data_bytes;
}

template <bool RawQ>
__device__ __forceinline__ void apply_packed_prefill_contract(FusedNormRopeAttnRopeCastParams& a)
{
    // The caller must first establish supports_packed_prefill on the host.
    a.q_norm                 = false;
    a.positions64            = 0;
    a.q_is_rotated           = !RawQ;
    a.out_ptr                = nullptr;
    a.debug_ptr              = nullptr;
    a.max_ptr                = nullptr;
    a.stride_kv_page         = packed_row_bytes;
    a.stride_kv_scale_page   = packed_row_bytes;
    a.stride_kv_extend       = packed_row_bytes;
    a.stride_kv_scale_extend = packed_row_bytes;
    a.unified_kv_scale_ptr   = static_cast<const char*>(a.unified_kv_ptr) + packed_data_bytes;
    a.kv_scale_ptr           = static_cast<const char*>(a.kv_ptr) + packed_data_bytes;
    __builtin_assume(a.attn_sink_ptr != nullptr);
    __builtin_assume(a.lse_ptr != nullptr);
}

} // namespace fused_norm_rope_attn_rope_cast_fwd
