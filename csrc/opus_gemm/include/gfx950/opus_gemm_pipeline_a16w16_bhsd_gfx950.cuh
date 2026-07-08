// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// BHSD-layout batch GEMM pipeline for a16w16 (bf16).
// Fork of the split-barrier a16w16 pipeline with BHSD addressing for matrix A.
// A is in [outer_batch, heads_per_group, seqlen, head_dim] layout; each tile_k
// step remaps (tile_k) -> (head_index, head_dim_offset) so loads stay
// contiguous within a single head.  No bias support.
#pragma once

// Pull in layout functions (make_layout_ga_noscale, etc.), CPOL macros,
// traits, and the base noscale kernel (different name, no ODR conflict).
#include "opus_gemm_pipeline_a16w16_gfx950.cuh"

// ============================================================================
// BHSD-layout a16w16 GEMM kernel
// ============================================================================

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2)
void gemm_a16w16_bhsd_kernel(opus_gemm_bhsd_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    const int grid_dim_x = opus::grid_size_x() / opus::block_size_x();
    int wgid = (opus::block_id_y() * grid_dim_x) + opus::block_id_x();
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div_constexpr(kargs.n, T::B_N);
    const int total_wgs = num_tiles_m * num_tiles_n;

    // HipKittens XCD swizzle (Algorithm 1) for joint L2+LLC cache reuse
    int tile_m_id, tile_n_id;
    if constexpr (T::NUM_XCD > 1) {
        constexpr int nXCD = T::NUM_XCD;
        constexpr int W = T::SWIZZLE_W;
        constexpr int C = T::SWIZZLE_C;
        int xy = wgid;
        int blocks_per_cycle = nXCD * C;
        int tid_per_group = W * num_tiles_n;
        int limit = (total_wgs / blocks_per_cycle) * blocks_per_cycle;
        if (xy >= limit) {
            int full_groups = limit / tid_per_group;
            int covered_cols = (limit - full_groups * tid_per_group) / W;
            int partial_first_row = full_groups * W;
            int partial_row_extent = num_tiles_m - partial_first_row;
            if (partial_row_extent > W) partial_row_extent = W;
            int f = xy - limit;
            int remaining_in_partial =
                (partial_row_extent > 0)
                    ? (num_tiles_n - covered_cols) * partial_row_extent
                    : 0;
            if (f < remaining_in_partial) {
                tile_m_id = partial_first_row + (f % partial_row_extent);
                tile_n_id = covered_cols + (f / partial_row_extent);
            } else {
                int g = f - remaining_in_partial;
                tile_m_id = (partial_first_row + partial_row_extent) + g / num_tiles_n;
                tile_n_id = g % num_tiles_n;
            }
        } else {
            int xcd = xy % nXCD;
            int local = xy / nXCD;
            int chunk_idx = local / C;
            int pos = local % C;
            int new_xy = xcd * C + chunk_idx * blocks_per_cycle + pos;
            int group_id = new_xy / tid_per_group;
            int first_row = group_id * W;
            int win_h = num_tiles_m - first_row;
            if (win_h > W) win_h = W;
            int l = new_xy % tid_per_group;
            tile_m_id = first_row + (l % win_h);
            tile_n_id = l / win_h;
            if (tile_n_id >= num_tiles_n) {
                tile_m_id = xy / num_tiles_n;
                tile_n_id = xy % num_tiles_n;
            }
        }
    } else {
        tile_m_id = wgid / num_tiles_n;
        tile_n_id = wgid % num_tiles_n;
    }
    int row = tile_m_id * T::B_M;
    int col = tile_n_id * T::B_N;

    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    // BHSD addressing: A base pointer uses stride_a_seq (= head_dim) as row stride.
    // Bound must span every head this tile may touch, not just head 0's row
    // range -- a_offset() below adds h * stride_a_head for h in
    // [0, heads_per_group). Sizing num_records to (kargs.m - row) * stride_a_seq
    // (one head's worth) silently OOB-zeros every h >= 1 access, corrupting
    // the result for heads_per_group > 1.
    //
    // Bound directly from the real (row, head, d) index ranges instead of
    // assuming any particular stride nesting: this callsite is fed both a
    // genuinely-contiguous BHSD tensor (stride_a_head = seqlen*stride_a_seq,
    // i.e. head is the 2nd-largest stride) *and* a strided BSHD view
    // (batch_gemm_a16w16_bshd_opus's bhsd_remap path, where stride_a_seq is
    // actually the LARGEST stride -- (kargs.batch - batch_id)*stride_a_batch
    // as a bound is far too small there and wrongly zeros real data). The
    // formula below is the exact offset of the last real element
    // (row=m-1, h=heads_per_group-1, d=head_dim-1) relative to this g_a's
    // base, so it can never exceed the real tensor allocation (no fault) for
    // any stride layout, while any M-tail / tile overshoot beyond it safely
    // OOB-zeros via the buffer descriptor -- those rows are already discarded
    // by the store-side HAS_OOB predicate.
    const int a_num_records_elems = (kargs.m - 1 - row) * kargs.stride_a_seq
                                   + (kargs.heads_per_group - 1) * kargs.stride_a_head
                                   + kargs.head_dim;
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a_seq,
        a_num_records_elems * sizeof(D_A));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b, (kargs.n - col) * kargs.stride_b * sizeof(D_B));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);

    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    // BHSD: use stride_a_seq instead of stride_a for the A global layout
    auto u_ga = make_layout_ga_noscale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a_seq);
    auto u_sa = make_layout_sa_noscale<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra_noscale<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_noscale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb_noscale<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_noscale<T>(lane_id, wave_id_n);

    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 4];
    smem<D_A> s_a[2][2] = {
        {make_smem(reinterpret_cast<D_A*>(smem_a)),
         make_smem(reinterpret_cast<D_A*>(smem_a + smem_a_byte))},
        {make_smem(reinterpret_cast<D_A*>(smem_a + 2 * smem_a_byte)),
         make_smem(reinterpret_cast<D_A*>(smem_a + 3 * smem_a_byte))}
    };
    constexpr int smem_b_byte = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 4];
    smem<D_B> s_b[2][2] = {
        {make_smem(reinterpret_cast<D_B*>(smem_b)),
         make_smem(reinterpret_cast<D_B*>(smem_b + smem_b_byte))},
        {make_smem(reinterpret_cast<D_B*>(smem_b + 2 * smem_b_byte)),
         make_smem(reinterpret_cast<D_B*>(smem_b + 3 * smem_b_byte))}
    };

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b[2];
    typename decltype(mma)::vtype_c v_c[2][2];
    clear(v_c[0][0]);
    clear(v_c[0][1]);
    clear(v_c[1][0]);
    clear(v_c[1][1]);

    // BHSD a_offset: remap tile_k to (head_index, head_dim_offset)
    auto a_offset = [&](int half_tile_m, int tile_k) {
        int k_abs = tile_k * T::B_K;
        int h = k_abs / kargs.head_dim;     // head_dim is power of 2, compiler optimizes
        int d = k_abs % kargs.head_dim;     // to shift/mask
        return h * kargs.stride_a_head
             + half_tile_m * T::HALF_B_M * kargs.stride_a_seq
             + d;
    };
    auto b_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
    };

    const int loops = ceil_div(kargs.k, T::B_K);
    int tic = 0, toc = 1;

    // Prologue
    async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, 0), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
    async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, 0), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
    async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, 0), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
    async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, 0), opus::number<0>{}, opus::number<T::CACHECTL_A>{});

    if (wave_id_m == 1) __builtin_amdgcn_s_barrier();

    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, 1), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
    async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
    async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, 1), opus::number<0>{}, opus::number<T::CACHECTL_B>{});

    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
    __builtin_amdgcn_s_barrier();

    // Main loop
    for(int tile = 0; tile < loops - 2; tile += 2) {
        // First tile
        v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
        async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
        async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        v_b[0] = load<T::VEC_B>(s_b[toc][0], u_rb);

        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Second tile
        v_a = load<T::VEC_A>(s_a[toc][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = load<T::VEC_B>(s_b[toc][1], u_rb);
        async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[toc][1], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);

        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    {
        int tile = loops - 2;

        v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);

        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1;
        toc ^= 1;
    }

    {
        v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
        v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    if (wave_id_m == 0) __builtin_amdgcn_s_barrier();

    // Store results to global memory with bounds checking and type conversion
    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    auto u_gc_m = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(1_I, 0_I), p_coord_c);
    auto u_gc_n = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(0_I, 1_I), p_coord_c);

    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * kargs.stride_c + half_tile_n * T::HALF_B_N;
    };

    // No bias support in BHSD kernel

    auto store_c = [&](auto& vc, int half_tile_m, int half_tile_n) {
        int g_c_offset = c_offset(half_tile_m, half_tile_n);
        int m_base = row + half_tile_m * T::HALF_B_M;
        int n_base = col + half_tile_n * T::HALF_B_N;

        if constexpr (T::HAS_OOB) {
            auto pred = [&](auto... ids) {
                return (m_base + u_gc_m(ids...)) < kargs.m && (n_base + u_gc_n(ids...)) < kargs.n;
            };
            if constexpr (std::is_same_v<D_C, D_ACC>) {
                store_if<T::VEC_C>(g_c, pred, vc, u_gc, g_c_offset, opus::number<CPOL_NT>{});
            } else {
                auto vc_out = cast<D_C>(vc);
                store_if<T::VEC_C>(g_c, pred, vc_out, u_gc, g_c_offset, opus::number<CPOL_NT>{});
            }
        } else {
            if constexpr (std::is_same_v<D_C, D_ACC>) {
                store<T::VEC_C>(g_c, vc, u_gc, g_c_offset, opus::number<CPOL_NT>{});
            } else {
                auto vc_out = cast<D_C>(vc);
                store<T::VEC_C>(g_c, vc_out, u_gc, g_c_offset, opus::number<CPOL_NT>{});
            }
        }
    };

    store_c(v_c[0][0], 0, 0);
    store_c(v_c[0][1], 0, 1);
    store_c(v_c[1][0], 1, 0);
    store_c(v_c[1][1], 1, 1);
#else
    // Non-gfx950 device pass: empty stub. See gfx950 branch above.
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
