// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <cstdint>

// Performance heuristics calibrated on MI355X/gfx950. These functions deliberately
// depend on workload shape; they are not correctness or cache-layout requirements.
// Capacity estimates allocated indices/query without reading device CSR contents.
namespace fused_norm_rope_attn_rope_cast_policy {

constexpr bool has_capacity(int tokens, int64_t capacity, int rows_per_query)
{
    return capacity >= int64_t(tokens) * rows_per_query;
}

constexpr bool prefer_h64_64row(int tokens, int64_t capacity, bool two_sources)
{
    return (tokens <= 256 && has_capacity(tokens, capacity, 512)) ||
           (tokens <= 1024 && has_capacity(tokens, capacity, 1536)) ||
           (two_sources && ((tokens <= 768 && has_capacity(tokens, capacity, 768)) ||
                            (tokens <= 1024 && has_capacity(tokens, capacity, 896)) ||
                            (tokens > 1024 && has_capacity(tokens, capacity, 1024))));
}

constexpr bool prefer_h128_64row(int tokens, int64_t capacity)
{
    return tokens <= 320 && has_capacity(tokens, capacity, 1024);
}

constexpr bool prefer_32head_tile(int tokens, int heads, int64_t capacity)
{
    // Double a 128-CTA grid to fill a 256-CU scheduling round for short lists.
    return int64_t(tokens) * (heads / 64) == 128 && !has_capacity(tokens, capacity, 768);
}

constexpr bool prefer_h128_shared_kv(int tokens, int64_t capacity, bool prepared_q)
{
    // The latter half of each 256-token band amortizes the 8-wave CTA tail.
    // At larger T, only the prepared-Q variant has demonstrated a gain.
    const bool token_band = (tokens > 320 && tokens <= 1024 && (tokens - 1) % 256 >= 128) ||
                            (prepared_q && tokens > 1024);
    return token_band && has_capacity(tokens, capacity, 1024);
}

constexpr bool prefer_h64_bounded_prefill(int tokens) { return tokens >= 512; }
constexpr bool prefer_h64_packed_prefill(int tokens) { return tokens >= 512 && tokens <= 1024; }
constexpr bool prefer_h128_late_bf16_load(int tokens) { return tokens >= 2048; }

} // namespace fused_norm_rope_attn_rope_cast_policy
