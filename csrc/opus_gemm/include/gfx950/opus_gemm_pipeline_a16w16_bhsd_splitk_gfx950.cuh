// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// BF16 a16w16 BHSD split-K pipeline: 4-wave warp-spec main kernel that
// writes fp32 partial sums into a workspace tensor. Fork of the flatmm
// split-K pipeline with BHSD addressing for the A operand.
//
// A is in BHSD layout [outer_batch, heads_per_group, seqlen, head_dim].
// The kernel remaps tile_k to (head_index, head_dim_offset) so that each
// tile_k load reads a contiguous segment within one head.
//
// The companion reduce kernel that sums across splits and casts to D_OUT
// (bf16 or fp32) C lives in the shared header splitk_reduce.cuh.
//
// Two kernels per launcher:
//   * gemm_a16w16_bhsd_splitk_kernel<Traits>(kargs)
//       grid.x = split_k * num_tiles_m * num_tiles_n
//       writes fp32 [split_k, B, padded_M, padded_N]
//   * splitk_reduce_kernel<VEC=16, BLOCK=64, D_OUT>(ws, c, ...)
//       defined in splitk_reduce.cuh; D_OUT picks the output cast.
#pragma once

#include "opus_gemm_traits_a16w16_gfx950.cuh"
#include "splitk_reduce_gfx950.cuh"

// Layout functions -- reuse from the original splitk pipeline.
// Brings make_layout_gmem_group_load_splitk, make_layout_smem_group_load_splitk,
// make_layout_ra_splitk, make_layout_rb_splitk, mask_va_tail.
#include "opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh"

// ============================================================================
// Main kernel: BHSD split-K variant.
// ============================================================================

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a16w16_bhsd_splitk_kernel(opus_gemm_bhsd_splitk_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    // gfx950-only kernel body. See opus_gemm_pipeline_a16w16_gfx950.cuh for the
    // multi-arch wheel rationale.
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    // grid.x = split_k * num_tiles_m * num_tiles_n (S splits fused to inner axis,
    // column-major tile walk for B reuse).
    int wgid_full = opus::block_id_x();
    int split_id  = wgid_full % kargs.split_k;
    int wgid      = wgid_full / kargs.split_k;
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    // Edge tile: partial-M or partial-N. OOB-zero buffer_loads on edge tiles
    // make the counted `s_waitcnt_vmcnt(number<...>{})` producer drains
    // unsafe -- vmem completion is not FIFO and the count may not include
    // slot 0 when the consumer crosses the barrier. Force full drain on
    // edge tiles only; aligned tiles keep the counted fast path.
    const bool tile_edge = (row + T::B_M > kargs.m) || (col + T::B_N > kargs.n);

    // K partitioning: chunk K into B_K-sized iters, distribute across splits.
    //   total_iters = ceil_div(K, B_K); iters_full = ceil_div(total_iters, split_k)
    //   First (split_k-1) splits each get `iters_full` full B_K iters (all in-range).
    //   Last split gets the remainder (may be less, final iter may be partial K).
    const int total_iters = ceil_div(kargs.k, T::B_K);
    const int iters_full  = ceil_div(total_iters, kargs.split_k);
    const int my_loops    = (split_id < kargs.split_k - 1)
                            ? iters_full
                            : (total_iters - (kargs.split_k - 1) * iters_full);
    const int k_start     = split_id * iters_full * T::B_K;

    // Edge case: if split_k > total_iters the last split ends up with my_loops <= 0.
    // Bail rather than write garbage to workspace.
    if (my_loops <= 0) return;

    const bool is_last_split        = (split_id == kargs.split_k - 1);
    const int  k_valid_in_last_iter = kargs.k - (total_iters - 1) * T::B_K;  // in [1, B_K]

    // BHSD addressing for A: decompose k_start into (head, offset_within_head).
    const int k_start_h = k_start / kargs.head_dim;
    const int k_start_d = k_start % kargs.head_dim;

    // Bound must span every head this split's K range may touch (h_delta can
    // run from 0 up to heads_per_group-1-k_start_h within a single split, e.g.
    // whenever split_k < heads_per_group), not just k_start_h's own row range
    // -- a_offset() below adds h_delta * stride_a_head on top of this base.
    // Sizing num_records to one head's worth (as (kargs.m - row) * stride_a_seq
    // - k_start_d used to) silently OOB-zeros every h_delta >= 1 access,
    // corrupting the result whenever a split spans more than one head (in
    // particular split_k == 1, i.e. the non-split-K-aware default).
    //
    // Bound directly from the real (row, head, d) index ranges instead of
    // assuming any particular stride nesting: this callsite is fed both a
    // genuinely-contiguous BHSD tensor (stride_a_head = seqlen*stride_a_seq)
    // *and* a strided BSHD view (batch_gemm_a16w16_bshd_opus's bhsd_remap
    // path, where stride_a_seq is actually the LARGEST stride -- a
    // (kargs.batch - batch_id)*stride_a_batch-based bound is far too small
    // there and wrongly zeros real data). The formula below is the exact
    // offset of the last real element (row=m-1, h=heads_per_group-1,
    // d=head_dim-1) relative to this g_a's base, so it can never exceed the
    // real tensor allocation (no fault) for any stride layout, while any
    // overshoot beyond it safely OOB-zeros via the buffer descriptor -- those
    // rows are discarded by the split-K reduce's own M/N bounds check
    // downstream.
    const int a_base_offset = k_start_h * kargs.stride_a_head
                             + row * kargs.stride_a_seq
                             + k_start_d;
    const int a_num_records_elems = (kargs.m - 1 - row) * kargs.stride_a_seq
                                   + (kargs.heads_per_group - 1 - k_start_h) * kargs.stride_a_head
                                   + kargs.head_dim - k_start_d;
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + batch_id * kargs.stride_a_batch
                         + a_base_offset,
                         a_num_records_elems * sizeof(D_A));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + batch_id * kargs.stride_b_batch + col * kargs.stride_b + k_start,
                         ((kargs.n - col) * kargs.stride_b - k_start) * sizeof(D_B));
    // Deref the handle slot at entry; survives a post-capture grow.
    D_C* ws_ptr = reinterpret_cast<D_C*>(kargs.ws_handle->ptr);
    auto g_c = make_gmem(ws_ptr
                         + (size_t)split_id  * kargs.batch * kargs.stride_ws_batch
                         + (size_t)batch_id  * kargs.stride_ws_batch
                         + (size_t)row       * kargs.stride_ws
                         + (size_t)col);

    // Warp specialization: role split happens up front. Role swaps every 256
    // workgroups so per-SIMD register layout stays balanced across CUs.
    int role = ((wave_id & 1) ^ ((wgid >> 8) & 1));

    // LDS layout: s_a[pfk][num_m_blocks][num_k_groups],
    //             s_b[pfk][num_n_blocks][num_k_groups].
    __shared__ char smem_a[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BM * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BN * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot_k, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    // BHSD a_offset: remap relative K offset to (head_delta, head_dim_offset).
    // g_a base already includes k_start_h * stride_a_head + k_start_d, so
    // offsets are relative to that base.
    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        int k_rel = (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
        int k_total = k_start_d + k_rel;
        int h_delta = k_total / kargs.head_dim;
        int d = k_total % kargs.head_dim;
        return h_delta * kargs.stride_a_head
             + group_load_idx * T::LOAD_GROUP_M * kargs.stride_a_seq
             + d - k_start_d;
    };
    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    const int loops = my_loops;

    constexpr int mb_a = T::a_buffer_load_insts;
    constexpr int mb_b = T::b_buffer_load_insts;
    constexpr int mb   = mb_a + mb_b;

    if (role == 0) {
        // ================= PRODUCER (2 waves) =================
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_splitk<T, 2>(lane_id, wave_id_prod, kargs.stride_a_seq);
        auto u_sa = make_layout_smem_group_load_splitk<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_group_load_splitk<T, 2>(lane_id, wave_id_prod, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_splitk<T, 2>(lane_id, wave_id_prod);

        // Prologue: fill pfk slots (K=0..pfk-1).
        opus::static_for<T::prefetch_k_iter>([&](auto p_c) {
            constexpr int p = decltype(p_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                    constexpr int m = decltype(m_c)::value;
                    async_load<T::VEC_A>(g_a, smem_a_at(p, m, kg), u_ga, u_sa, a_offset(p, m, kg));
                });
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(p, n, kg), u_gb, u_sb, b_offset(p, n, kg));
                });
            });
        });

        // Prologue barriers R_0..R_{pfk-3}: leave 2*mb in flight for depth-2 pipeline entry.
        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = T::prefetch_k_iter - 1 - decltype(i_c)::value;
            if (tile_edge) s_waitcnt_vmcnt(0_I);
            else           s_waitcnt_vmcnt(number<mb * p>{});
            __builtin_amdgcn_s_barrier();
        });

        if constexpr (T::prefetch_k_iter == 3) {
            // Depth-1 pipeline for pfk=3.
            if (tile_edge) s_waitcnt_vmcnt(0_I);
            else           s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();   // R_1
            for (int i = T::prefetch_k_iter - 1; i < loops - 1; i++) {
                int issue_k = i + 1;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                    });
                });
                if (tile_edge) s_waitcnt_vmcnt(0_I);
                else           s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
            }
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();   // R_{loops-1}
        }
        else {
            // Depth-2 pipeline for pfk>=4.
            for (int i = T::prefetch_k_iter - 2; i < loops - 2; i++) {
                int issue_k = i + 2;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                    });
                });
                if (tile_edge) s_waitcnt_vmcnt(0_I);
                else           s_waitcnt_vmcnt(number<2 * mb>{});
                __builtin_amdgcn_s_barrier();
            }

            // Epilogue: drain the last two in-flight K's.
            if (tile_edge) s_waitcnt_vmcnt(0_I);
            else           s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
    }
    else {
        // ================= CONSUMER (2 waves) =================
        int wave_id_m = wave_id / 2;
        int wave_id_n_cons = 0;
        auto u_ra = make_layout_ra_splitk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_splitk<T>(lane_id);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a0, v_a1;
        typename decltype(mma)::vtype_b v_b0, v_b1;
        typename decltype(mma)::vtype_c v_c;
        clear(v_c);

        constexpr int ds_read_insts = T::a_ds_read_insts + T::b_ds_read_insts;

        // Prologue Step 0: R_0 -> load K=0.
        __builtin_amdgcn_s_barrier();
        {
            auto sa0 = make_smem(smem_a_at(0, 0, 0));
            auto sb0 = make_smem(smem_b_at(0, 0, 0));
            v_a0 = load<T::VEC_A>(sa0, u_ra);
            v_b0 = load<T::VEC_B>(sb0, u_rb);
        }

        // Prologue Steps 1..pfk-2.
        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = decltype(i_c)::value + 1;
            constexpr int cur = (p - 1) & 1;
            constexpr int nxt = p & 1;
            __builtin_amdgcn_s_barrier();
            auto sa_p = make_smem(smem_a_at(p, 0, 0));
            auto sb_p = make_smem(smem_b_at(p, 0, 0));
            if constexpr (nxt == 0) {
                v_a0 = load<T::VEC_A>(sa_p, u_ra);
                v_b0 = load<T::VEC_B>(sb_p, u_rb);
            } else {
                v_a1 = load<T::VEC_A>(sa_p, u_ra);
                v_b1 = load<T::VEC_B>(sb_p, u_rb);
            }
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            if constexpr (cur == 0) v_c = mma(v_a0, v_b0, v_c);
            else                    v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);
        });

        // L = (pfk - 2) & 1: compile-time buf where the live tile sits
        // entering the main loop. Main loop alternates v_a0/v_a1 per half-iter.
        constexpr int L = (T::prefetch_k_iter - 2) & 1;

        int k = T::prefetch_k_iter - 1;
        for (; k + 1 < loops - 1; k += 2) {
            // half-iter A
            __builtin_amdgcn_s_barrier();
            {
                int slot = k % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            if constexpr (L == 0) v_c = mma(v_a0, v_b0, v_c);
            else                  v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);

            // half-iter B
            __builtin_amdgcn_s_barrier();
            {
                int slot = (k + 1) % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            if constexpr (L == 0) v_c = mma(v_a1, v_b1, v_c);
            else                  v_c = mma(v_a0, v_b0, v_c);
            __builtin_amdgcn_s_setprio(0);
        }

        // Tail half-iter if (loops - pfk) is odd.
        bool last_in_buf1 = (L != 0);
        if (k < loops - 1) {
            __builtin_amdgcn_s_barrier();
            {
                int slot = k % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                if constexpr (L == 0) {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            if constexpr (L == 0) v_c = mma(v_a0, v_b0, v_c);
            else                  v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);
            last_in_buf1 = (L == 0);
            k++;
        }

        // Epilogue: load K=loops-1, mma K=loops-2 then K=loops-1.
        // Last mma's v_a may contain K beyond kargs.k (last split's tail), so
        // apply the mask_va_tail zeroing only for the final MMA.
        __builtin_amdgcn_s_barrier();
        int last_slot = (loops - 1) % T::prefetch_k_iter;
        auto sa_last = make_smem(smem_a_at(last_slot, 0, 0));
        auto sb_last = make_smem(smem_b_at(last_slot, 0, 0));
        const int k_valid_final = is_last_split ? k_valid_in_last_iter : T::B_K;
        if (last_in_buf1) {
            v_a0 = load<T::VEC_A>(sa_last, u_ra);
            v_b0 = load<T::VEC_B>(sb_last, u_rb);
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);
            s_waitcnt_lgkmcnt(0_I);
            if constexpr (T::HAS_OOB) { mask_va_tail<T>(v_a0, k_valid_final, lane_id); }
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a0, v_b0, v_c);
            __builtin_amdgcn_s_setprio(0);
        } else {
            v_a1 = load<T::VEC_A>(sa_last, u_ra);
            v_b1 = load<T::VEC_B>(sb_last, u_rb);
            s_waitcnt_lgkmcnt(number<ds_read_insts>{});
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a0, v_b0, v_c);
            __builtin_amdgcn_s_setprio(0);
            s_waitcnt_lgkmcnt(0_I);
            if constexpr (T::HAS_OOB) { mask_va_tail<T>(v_a1, k_valid_final, lane_id); }
            __builtin_amdgcn_s_setprio(1);
            v_c = mma(v_a1, v_b1, v_c);
            __builtin_amdgcn_s_setprio(0);
        }

        // Store to workspace (fp32, padded_M x padded_N tile-aligned -> no pred).
        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
                        opus::make_tuple(kargs.stride_ws, 1_I), p_coord_c);
        store<T::VEC_C>(g_c, v_c, u_gc, 0);
    }
#else
    // Non-gfx950 device pass: empty stub. See gfx950 branch above.
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}
