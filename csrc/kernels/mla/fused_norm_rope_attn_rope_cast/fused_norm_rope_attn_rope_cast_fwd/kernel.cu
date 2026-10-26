// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "../dispatch_policy.hpp"
#include "decode.cuh"
#include "kernel.cuh"
#include "kernel.h"
#include "prefill_contract.hpp"

template <int Heads,
          int CacheMode,
          bool RawQ,
          bool Continuous     = false,
          bool DirectQ        = false,
          int HeadTile        = 64,
          bool BoundedPrefill = false,
          bool PackedPrefill  = false>
__global__ __launch_bounds__(256, 1) void fused_norm_rope_attn_rope_cast_fwd_kernel(
    FusedNormRopeAttnRopeCastParams args)
{
    fused_norm_rope_attn_rope_cast_fwd::specialize_heads<Heads>(args);
    if constexpr(PackedPrefill)
    {
        static_assert(Heads == 64 && CacheMode == 0 && BoundedPrefill);
        fused_norm_rope_attn_rope_cast_fwd::apply_packed_prefill_contract<RawQ>(args);
    }
    using Config = FusedNormRopeAttnRopeCastConfig<CacheMode,
                                                   RawQ,
                                                   DirectQ,
                                                   0,
                                                   Heads,
                                                   HeadTile,
                                                   BoundedPrefill,
                                                   PackedPrefill>;
    __shared__ __align__(16) char smem[Config::smem_bytes() + 2 * 68 * sizeof(int)];
    if(indices_valid<Config>(args))
        fused_norm_rope_attn_rope_cast_fwd_impl<Config, false, Continuous>(args, smem);
    else
        fused_norm_rope_attn_rope_cast_fwd_impl<Config, true>(args, smem);
}

template <bool PackedPrefill>
inline hipError_t launch_h64_bounded_prefill(const FusedNormRopeAttnRopeCastParams& args,
                                             bool raw_q,
                                             hipStream_t stream)
{
    const dim3 grid(args.N);
    if(raw_q)
        fused_norm_rope_attn_rope_cast_fwd_kernel<64, 0, true, true, true, 64, true, PackedPrefill>
            <<<grid, 256, 0, stream>>>(args);
    else
        fused_norm_rope_attn_rope_cast_fwd_kernel<64,
                                                  0,
                                                  false,
                                                  true,
                                                  false,
                                                  64,
                                                  true,
                                                  PackedPrefill><<<grid, 256, 0, stream>>>(args);
    return hipGetLastError();
}

template <int Heads>
hipError_t run_fused_norm_rope_attn_rope_cast_heads(const FusedNormRopeAttnRopeCastParams& args,
                                                    bool raw_q,
                                                    hipStream_t stream)
{
    namespace policy  = fused_norm_rope_attn_rope_cast_policy;
    namespace prefill = fused_norm_rope_attn_rope_cast_fwd;
    const dim3 grid(args.N, Heads / 64);
    if(args.num_splits > 1)
    {
        const dim3 split_grid(args.N, Heads / 64, args.num_splits);
        if(args.num_splits == 2)
        {
            if(raw_q)
                fused_norm_rope_attn_rope_cast_decode_kernel<Heads, true, true, 2>
                    <<<split_grid, 256, 0, stream>>>(args);
            else
                fused_norm_rope_attn_rope_cast_decode_kernel<Heads, false, false, 2>
                    <<<split_grid, 256, 0, stream>>>(args);
        }
        else if(args.num_splits == 4)
        {
            if(raw_q)
                fused_norm_rope_attn_rope_cast_decode_kernel<Heads, true, true, 4>
                    <<<split_grid, 256, 0, stream>>>(args);
            else
                fused_norm_rope_attn_rope_cast_decode_kernel<Heads, false, false, 4>
                    <<<split_grid, 256, 0, stream>>>(args);
        }
        else if(raw_q)
            fused_norm_rope_attn_rope_cast_decode_kernel<Heads, true, true>
                <<<split_grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_decode_kernel<Heads, false>
                <<<split_grid, 256, 0, stream>>>(args);
        const auto status = hipGetLastError();
        if(status != hipSuccess)
            return status;
        if(args.num_splits == 2)
            fused_norm_rope_attn_rope_cast_decode_reduce_kernel<Heads, 2>
                <<<dim3(args.N, Heads / 16), 256, 0, stream>>>(args);
        else if(args.num_splits == 4)
            fused_norm_rope_attn_rope_cast_decode_reduce_kernel<Heads, 4>
                <<<dim3(args.N, Heads / 16), 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_decode_reduce_kernel<Heads>
                <<<dim3(args.N, Heads / 16), 256, 0, stream>>>(args);
        return hipGetLastError();
    }
    if(args.cache_mode == 0)
    {
        if(args.prefix_capacity > 0 && args.extend_capacity > 0)
        {
            const int64_t capacity = int64_t(args.prefix_capacity) + args.extend_capacity;
            if(policy::prefer_32head_tile(args.N, Heads, capacity))
            {
                const dim3 head_grid(args.N, Heads / 32);
                if(raw_q)
                    fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 0, true, true, true, 32>
                        <<<head_grid, 256, 0, stream>>>(args);
                else
                    fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 0, false, true, false, 32>
                        <<<head_grid, 256, 0, stream>>>(args);
                return hipGetLastError();
            }
            if constexpr(Heads == 64)
            {
                if(policy::prefer_h64_bounded_prefill(args.N) && prefill::kv_offsets_fit_u32(args))
                {
                    if(policy::prefer_h64_packed_prefill(args.N) &&
                       prefill::supports_packed_prefill(args, raw_q))
                        return launch_h64_bounded_prefill<true>(args, raw_q, stream);
                    return launch_h64_bounded_prefill<false>(args, raw_q, stream);
                }
            }
            if(raw_q)
                fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 0, true, true, Heads == 64>
                    <<<grid, 256, 0, stream>>>(args);
            else
                fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 0, false, true>
                    <<<grid, 256, 0, stream>>>(args);
        }
        else if(raw_q)
            fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 0, true, false, Heads == 64>
                <<<grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 0, false>
                <<<grid, 256, 0, stream>>>(args);
    }
    else if(args.cache_mode == 1)
    {
        if(raw_q)
            fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 1, true>
                <<<grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 1, false>
                <<<grid, 256, 0, stream>>>(args);
    }
    else
    {
        if(raw_q)
            fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 2, true>
                <<<grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_fwd_kernel<Heads, 2, false>
                <<<grid, 256, 0, stream>>>(args);
    }
    return hipGetLastError();
}

hipError_t run_fused_norm_rope_attn_rope_cast_fwd(const FusedNormRopeAttnRopeCastParams& args,
                                                  bool raw_q,
                                                  hipStream_t stream)
{
    if(args.H == 64)
        return run_fused_norm_rope_attn_rope_cast_heads<64>(args, raw_q, stream);
    if(args.H == 128)
        return run_fused_norm_rope_attn_rope_cast_heads<128>(args, raw_q, stream);
    return hipErrorInvalidValue;
}
