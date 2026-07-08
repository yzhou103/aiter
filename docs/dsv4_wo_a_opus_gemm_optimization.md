# DeepSeek-V4 输出投影(wo_a)opus GEMM 优化调查记录

对象:DeepSeek-V4 attention output-LoRA GEMM(`ATOM/atom/models/deepseek_v4.py`
里的 `o = torch.einsum("sgd,grd->sgr", o, wo_a)` / `batched_gemm_bf16` 分支)。
目标:用 aiter 的 opus a16w16 GEMM 替换,并尽量逼近 / 超过 hipBLASLt。
所有测量在 **MI355X (gfx950)** 上,`HIP_VISIBLE_DEVICES=0`(确认 idle)。

DeepSeek-V4 形状:`o=[num_tokens, n_local_groups=8, K=4096]`,
`wo_a=[8, o_lora_rank=1024, 4096]`,输出 `[num_tokens, 8, 1024]`。
映射到 GEMM:batch=8, M=num_tokens, N=1024, K=4096。

---

## 0. 关键纪律

- 测性能一律用 idle GPU + `rocprofv3 --kernel-trace`(真实 GPU 时间);
  wall-clock 只作参考。
- ATT(`rocprofv3 --att`)在本环境需要手动装 `librocprof-trace-decoder.so`
  (ROCm 7.2.x 二进制包不带,仓库根目录的
  `rocprof-trace-decoder-ubuntu-22.04-0.1.2-Linux.deb` 里有,`dpkg-deb -x`
  取出 `.so` 放进 `/opt/rocm/lib` 即可)。
- ATT 单 CU 的 stall 绝对值不能跨 kernel 直接比(不同 BLOCK_SIZE 同 CU 上
  驻留的 wave 数不同,stall 是跨并发 wave 累加的),只看**占比结构**,快慢
  以 wall-clock / kernel-trace 为准。

---

## 1. 已落地的改动(均过完整回归,err=0)

1. **BHSD/BSHD kernel 的 A buffer-bound 跨 head 清零 bug**
   `opus_gemm_pipeline_a16w16_bhsd(_splitk)_gfx950.cuh` 里 A 的
   `make_gmem` num_records 原本照搬标准 GEMM 的 `(m-row)*stride_a_seq`,
   只覆盖一个 head;BHSD 里 head 间距是 `stride_a_head`,跨 head 的 load
   越界被硬件清零 → `splitK=0` 时结果全错(err≈1.0)。改成按真实最后一个
   元素下标 `(m-1-row)*stride_a_seq + (hpg-1)*stride_a_head + head_dim`
   反推 bound,对 native BHSD 和 BSHD 转置视图两种 stride 排布都成立。

2. **kid 分类路由 + 强制编译**
   `opus_kid_is_splitk` / `kid_is_splitk` 补上 bhsd_splitk 段(600–649,
   +1000 nooob),否则 kid 608 被误判为非 splitk、路由进只有 fp32 实例的
   bf16 表报错。608/1608/650/1650 加进 `HEURISTIC_DEFAULT_KIDS_GFX950`
   强制编译集(否则子集编译不实例化它们)。

3. **splitk_reduce kernel 加 `stride_c` / `stride_c_batch` 参数**
   reduce kernel 原本硬编码 `c_idx = b*M*N + m*N + n`,只能写连续
   `[batch,M,N]`。加两个尾参(-1 哨兵 = 回退到旧行为,现有调用方零影响)
   后可直接写进 strided Y 视图,消掉 bshd 路径的 transpose-copy kernel。
   gfx950 / 942 / 1250 三套 reduce 签名 + 显式实例化 + 前向声明同步更新。

4. **新增 `wo_a_gemm_opus` + `_mmajor` launcher**
   DeepSeek-V4 output-LoRA 直接吃 `[T,G,K]` 原生布局,Python 侧
   **零 `.transpose()`/`.permute()`/`.contiguous()`**。transpose 融进
   launcher:`_mmajor` 变体读 XQ/Y 时 dim0=M、dim1=batch(常规 launcher 是
   dim0=batch、dim1=M),等价于把"哪个轴是 batch"的判断从 Python 挪进
   C++ launcher。为 `a16w16_flatmm_splitk`(kid 200 系,fp32-only)和
   `a16w16` split-barrier(kid 4–9,bf16/fp32)两族都生成了 `_mmajor`
   变体,复用同一份 `__global__` kernel,不产生新设备代码。

5. **默认 kid 按 num_tokens 二选一**
   `wo_a_gemm_opus` 默认:`T>=300` 用 kid 9,否则 kid 208(见 §2/§4)。

改动文件:`csrc/opus_gemm/opus_gemm.cu`、`include/opus_gemm.h`、
`include/gfx950/opus_gemm_arch_gfx950.cuh`、`include/gfx950/splitk_reduce_gfx950.cuh`
(+942/1250)、`include/gfx950/opus_gemm_pipeline_a16w16_bhsd(_splitk)_gfx950.cuh`、
`codegen/gen_instances_gfx950.py`、`gen_instances.py`、`opus_gemm_common.py`、
`csrc/include/rocm_ops.hpp`、`csrc/pybind/opus_gemm_pybind.cu`、
`aiter/ops/opus/gemm_op_a16w16.py`。

---

## 2. 性能:kid 9 vs hipBLASLt(torch.einsum)

DeepSeek-V4 形状,wall-clock(us):

| num_tokens | kid 9 | einsum(hipBLASLt) | 差距 |
|---:|---:|---:|---:|
| 64   | 72.0  | 21.3  | 3.4x(小 T 走 kid 208)|
| 256  | 76.5  | 41.7  | 1.84x |
| 512  | 78.4  | 46.0  | 1.70x |
| 1024 | 83.2  | 60.0  | 1.39x |
| 2048 | 117.9 | 90.9  | 1.30x |
| 4096 | 235.9 | 199.8 | 1.18x |
| 8192 | 456.5 | 394.8 | 1.16x |

- 大 T 下 kid 9 落后 hipBLASLt **~1.15–1.4x,T 越大越接近**。
- 相比改造前的默认 kid 208(T=1024 时 162us),kid 9(~90us)**快 40–56%**。
- 小 T(<300)kid 9 的大 tile 填不满,默认切回 kid 208。

kernel-trace 分解(T=1024,单次调用):

| kernel | 数量 | 说明 |
|---|---|---|
| `gemm_a16w16_kernel`(kid 9) | 1 | ~90–98us,单 kernel 直写 Y |
| ~~transpose-copy~~ | 0 | 已被 §1.4 消除 |
| einsum 的 `Cijk_...`(hipBLASLt) | 1 | ~61us |

---

## 3. 为什么换 kid 9(实现对比)

试遍了代码库所有现成 a16w16 架构(T=1024):

| kid | 家族 | BLOCK / tile | wall |
|---:|---|---|---:|
| 208 | flatmm_splitk(warp-spec + splitK) | 256 / 64×64×128 | 138–162us |
| 4/5 | split-barrier | 256 / 128×256 等 | 118–125us |
| 6/8 | split-barrier | 512 / 128×256 | 110us |
| 300/303/315 | persistent | 512 / 256×256 | 90–101us |
| **9** | **split-barrier(非 splitk)** | **512 / 256×256** | **90us** |

kid 9 赢 kid 208 的两个核心原因:
1. **没有 producer/consumer 分工** → 不用付角色切换那个 3000+ cycle 的
   冷启动 barrier;
2. **单 kernel 直写 Y** → 不需要 splitK 的 fp32 workspace + reduce 第二趟。

---

## 4. kid 9 vs kid 208 结构对比

| | kid 9 | kid 208 |
|---|---|---|
| 结构 | 8 wave 全员搬+算,单 kernel 直写 Y | 生产者/消费者分工,split-K 双 kernel |
| tile / BLOCK | 256×256 / 512(8 wave) | 64×64×128 / 256(4 wave) |
| MFMA | 16×16×32 bf16 | 16×16×32 bf16 |
| 强项 | 大 T(大 tile 摊薄同步) | 小 T(小 tile 不浪费 + splitK 喂满) |
| 弱项 | 小 T tile 填不满 | 大 T 冷启动 barrier × 海量 WG + reduce 开销 |

kid 208 额外机制:`role = (wave_id&1) ^ ((wgid>>8)&1)` 把 4 wave 劈成
2 生产 + 2 消费;主 kernel 写 fp32 workspace `[split_k,B,padded_M,padded_N]`,
再由 `splitk_reduce_kernel` 沿 split_k 求和 + 转 dtype 写 Y。

---

## 5. kid 9 的 tile 分解与 AGPR 天花板

配置 `_a16w16(512,256,256,64, 4, 16,16,32)`:BLOCK_SIZE=512(8 wave),
B_M=B_N=256,B_K=64,T_M=2,T_N=4,MFMA 16×16×32,VEC_A/B=8,VEC_C=4。

三层切分(256×256 输出 tile):
1. **2×2 象限**(`v_c[2][2]`,HALF_B_M=HALF_B_N=128):每个 wave 参与全部
   4 个象限(循环维度)。
2. **象限内 8 wave 铺 2×4 网格**:`wave_id_m = wave_id/T_N`(0–1),
   `wave_id_n = wave_id%T_N`(0–3)。
3. **wave 在单象限内覆盖 E_M×E_N 个 16×16 MFMA 块**:
   `E_M = 128/(16·2) = 4`,`E_N = 128/(16·4) = 2`。

每个 wave 持有的 16×16 输出块 = `4 象限 × E_M×E_N(=8) = 32`;
全局校验 `32 × 8 wave = 256 = (256/16)²` ✓。

**AGPR 推导**:单个 16×16 fp32 输出块 = 256 结果 ÷ 64 lane = 4 个 AGPR/lane
(即 `W_M·W_N/64`)。
```
AGPR/lane = 4(象限) × E_M(4) × E_N(2) × (W_M·W_N/64 = 4) = 128
```
gfx950 每 lane AGPR 上限 256,128 恰好一半。

**天花板**:B_M 或 B_N 翻倍 → E_M 或 E_N 翻倍 → AGPR = 256,超过
codegen 的 `AGPR < 256` 检查,编译直接拒。所以 **256×256 是这套
16×16 MFMA + T_M=2,T_N=4 排布能开的最大 tile**。

**BLOCK_SIZE=256 更差(已实测,非推测)**:kid 4/5 就是 BS=256,
118–125us > kid 9 的 90us。而且 BS=256 无法配 256×256(4 wave 每 wave
扛双倍累加器 → AGPR=256 爆),被迫用小 tile → WG 数变多、复用率低。
**kid 9 的 512 是"能开最大 tile"和"占用率够"之间的最优点。**

---

## 6. 布局函数(为什么 LDS 跳不掉)

A 走三步,每步一个 `make_layout_*`(B 对称):
- `make_layout_ga_noscale`:全局→LDS,lane 拆 `lane%threads_k`(K 连续块,
  DRAM 合并友好)+ `lane/threads_k`(M 行)。
- `make_layout_sa_noscale`:写 LDS,行距带 `smem_padding` 错开 bank。
- `make_layout_ra_noscale`:LDS→寄存器,lane 拆法**完全不同**,按
  `v_mfma_f32_16x16x32_bf16` 硬布局:**行 = lane%16,K 组 = lane/16
  (4 组 × 8 连续 K = 32)**,每 lane 持 8 个 A 元素(= VEC_A)。

读写两端 lane→(m,k) 映射不同,LDS 是转接站:按①对 DRAM 友好灌进去,
按③对 MFMA 友好取出来。跳过 LDS 直读全局 → 合并访问垮掉,得不偿失。

`ra` 把行号 `lane%16` 再拆成 `%T_N` 和 `/T_N`:不是 MFMA 要求,而是
**复现 `sa` 写入时的 T_N 交错地址**(A 装载由 512 线程按 2×4 分工,
LDS 地址含 T_N 分量;读回必须同样拆解才命中原地址)。codegen 的
`T_M 必须=2`、`B_K = T_N·W_K/2` 就是这套交错自洽的代数条件。

---

## 7. ATT 三方 stall 画像(同一 GEMM,单 CU 稳态占比)

| stall 类别 | kid 208 | kid 9 | hipBLASLt |
|---|---:|---:|---:|
| MFMA(算) | 36.8% | 35.9% | 31.6% |
| **s_barrier(同步)** | **22.9%** | **22.1%** | **8.1%** |
| ds_read(LDS 读) | 21.2% | 12.2% | 15.4% |
| s_waitcnt(等访存) | 1.9% | 8.3% | 19.4% |
| buffer_load(全局读) | 0.4% | 7.3%* | 16.7% |
| ds_write(LDS 写) | — | — | 7.3% |
| SALU(地址/控制) | 14.6% | 3.6% | 0.1% |
| VALU | 1.5% | 2.9% | 1.1% |

\* kid 9 是 `buffer_load...lds`(async 直搬)。

解读:
- **hipBLASLt** 时间花在物理下限(waitcnt+load=36% 等/搬真实数据),
  barrier 仅 8%、SALU 0.1%(几乎无人为协调开销)——干净流水线。
- **kid 208** 病灶:barrier 22.9%(生产/消费握手)+ SALU 14.6%
  (splitK 分片索引 + 双组簿记)+ ds_read 21.2%(小 tile 多次小读),
  近 59% 是协调/搬运的额外成本。
- **kid 9** 相对 208:SALU 14.6→3.6%(无 splitK 索引/双组簿记)、
  ds_read 21.2→12.2%(大 tile 复用率高);但 **barrier 仍 22.1%** ——
  8 wave 共享同一 LDS tile,每切 buffer 都要全员 barrier。

**共同短板 = ~22% barrier**(warp-spec 握手 / 8-wave 共享 LDS),
正是 hipBLASLt 用更轻同步(8%)避开的部分。

---

## 8. 结论:现有框架已到顶

四条独立证据证明"在现有 pipeline 内调参"已到极限:
1. tile > 256×256 → AGPR ≥ 256 硬限,编译拒;
2. 减少协作 wave(T_M=1)→ pipeline 寄存器排布约束不允许,且能达到的
   最小配置(4 wave / BS=256)已实测更慢;
3. 32×32×8 MFMA:单指令 MACs 与 16×16×32 完全相同(都是 8192),无理论
   优势,且 gfx950 无实现;
4. ISA 级反汇编 kid 9:16 条 MFMA 背靠背、累加器错开,排布已紧凑,
   无明显可填空当。

**kid 9(16×16×32,256×256,8 wave)是现有代码库在 gfx950 上的最优。**
要真正追平 hipBLASLt,唯一剩下的路是写一个**减少跨 wave 共享/同步**的
全新结构 kernel(把那 22% barrier 降下来),风险与不确定性都显著更高,
应作为独立任务规划。

---

## 追加(2026-07-07):占用率实测 + 单-barrier 证伪

### grid/block:hipBLASLt vs opus(kernel-trace 元数据)
| | hipBLASLt | kid 9(opus)|
|---|---|---|
| block | `256×1×1`(4 wave)| `512×1×1`(8 wave)|
| grid | `65536×1×1`(**扁平 1D**,batch 拍进线性 ID)| `(num_tiles,1,batch=8)`(**batch 在 grid.z**)|

hipBLASLt 的 65536 ≈ 输出 tile 数(256)× 256,是 **Stream-K** 签名:超发扁平
grid,每个 WG 从线性 ID 反算 `(batch,m,n,k-range)`,K 维切开协作归约。opus 是朴
素数据并行,一个 WG 独包一个输出 tile 的整条 K。

### 占用率实测(rocprofv3 kernel-trace,不再靠 wall 猜)
| kid | tile / wave | LDS | block | LDS 限制 WG/CU |
|---|---|---|---|---|
| 9 / 40 | 256×256 / 8w | 132KiB | 512 | 1 |
| 41 | 256×128 / 4w | **49KiB** | 256 | **3** |
| 42 | 128×128 / 4w | 33KiB | 256 | 4 |

**关键**:kid 41 的占用率画像(49KiB、3 WG/CU、4 wave、block256)≈ hipBLASLt
(50KiB、3 WG/CU、4 wave),但仍 123us(hipBLASLt 71us)。→ **占用率不是瓶颈**;
之前"小 tile 慢 = 复用不够"的说法**被推翻**。

### ATT kid 41:barrier 40.6%
s_barrier **40.6%** / s_waitcnt 26% / MFMA 23.6%。小 tile 每单位计算 barrier 更密,
barrier 占比比 kid 9(22%)更高。矛头精确指向"减 barrier"。

### 单-barrier 重写(Phase 1 核心)——**实测更慢,假设证伪**
把 interleave 主循环改成每 K 步仅 1 个 barrier + 全量 `vmcnt(0)/lgkmcnt(0)` 排空:
- 结果:kid40 127us、kid41 165us、kid42 170us(全 err=0),**全部更慢**。
- 原因:原 kernel 的多 barrier 与**计数式部分等待**(`vmcnt(number<...>)`、
  `lgkmcnt(number<a_ds_read_insts>)`)一体,构成 ds_read/mma/async_load 交错的
  **细粒度流水**;换成全量排空后 ds_read 延迟被完全暴露(4 read 发完等全部才 mma),
  粒度变粗净更慢。**barrier 数不是可独立削减的杠杆**,ATT 的 40% 具误导性。

**结论更新**:Phase 1"单 barrier"路线经实测行不通;原调度对其结构已近最优。要更快
只能完整复刻 hipBLASLt(细粒度计数等待 + 高占用 + Stream-K 三者一体),即从零重写整
条流水,作为独立高风险任务。interleave header 已回退为 base 的精确克隆(kid40≈kid9,
err=0),新 kid 家族保留作实验底座。

---

## 追加(2026-07-07 下午):persistent 家族 + batch 维

### CU 占用真相(MI355X = 256 CU)
kid 9(256×256)全 shape 只有 `4×4×8=128` 个 tile → 只发 128 WG → **半块 GPU(128 CU)空转**,
且 LDS 132KiB 每 CU 只塞 1 WG,补不回来。hipBLASLt 256×128 → 256 tile = 256 WG,正好填满。

### persistent 家族(kid 300..303)——"高效 split-barrier 流水 + 填满 CU"的组合
persistent kernel = kid 9 那套细粒度 split-barrier 内层循环 + 持久化外层(小 tile → 更多 WG)。
**已内建 batch 支持**(kernel 用 `block_id_z()` + `stride_*_batch`,launcher 读 `XQ.size(0)`
设 `grid.z=batch`)。原本只有 300/1300 强编译;已把 301/302/303(+1000 nooob)加入
`HEURISTIC_DEFAULT_KIDS`。

单 GEMM(1024²×4096,batch=1):
| kid | tile | 时间 | vs kid9 |
|---|---|---|---|
| 9 / 300 | 256×256 | 76 / 74us | — |
| 301 | 128×256 | 47us | 1.6x |
| 302 | 256×128 | 50us | 1.5x |
| **303** | **128×128** | **35.8us** | **2.1x** |
| hipBLASLt | | 17.9us | |

batched(batch=8,真实 wo_a shape,kernel-only):
| kid | 时间 |
|---|---|
| hipBLASLt | 73us |
| **303(128×128)** | **89.4us**(opus 最快,1.23x)|
| 301 | 91.8us |
| 9 | 100us |
| 302 | 117.6us |

### 但端到端有转置税
persistent 要 `XQ=[G,T,K]` 连续;模型里 `o=[T,G,K]`,`transpose(0,1).contiguous()` 单独就
**46us**,使 persistent303 端到端 = **144us**,反而不如 kid 9 零拷贝(mmajor)的 98us。

**结论**:persistent 的 batch 维不用加(已有)。真正缺的是 **mmajor(融合转置)persistent
launcher**——按 `[T,G,K]` 的 stride 填 kargs(batch=G 在中间轴、M=T 在 0 轴),零拷贝。

### ✅ 已实现:persistent mmajor(零拷贝)launcher
改动(不动 kernel):
1. `codegen/gen_instances_gfx950.py:gen_persistent_instance` 追加 `{name}_mmajor` launcher
   emit + host 实例注册(strides 交换:`stride_a=XQ.stride(0)`,`stride_a_batch=XQ.stride(1)`,
   `stride_c=Y.stride(0)`,`stride_c_batch=Y.stride(1)`)。
2. `gen_instances.py`:把 `a16w16_persistent` 加进 `GENERATE_A16W16_TUNE_LOOKUP_MMAJOR_{FP32,BF16}`
   和 manifest 的 mmajor tag 集,使 persistent kid 可经 `wo_a_gemm_opus` 零拷贝分发。
3. `opus_gemm_common.py`:强编译 301/302/303(+1000 nooob)。

端到端(零拷贝,从模型原生 `[T,G,K]`,**`run_perftest` 稳定计时**,MI355X)。
⚠️ 早先用 `time.perf_counter` 的读数噪声大(误报 kid303 在 T=1024 快 11%);下表为稳定值:

| T | kid208 | kid9 | **kid301(128×256)** | kid303(128×128)| hipBLASLt |
|---|---|---|---|---|---|
| 64 | **28.9** | 74.5 | 60.9 | 65.5 | 22.1 |
| 128 | **46.3** | 76.9 | 67.4 | 79.3 | 26.3 |
| 256 | **53.9** | 84.8 | 68.2 | 77.0 | 42.0 |
| 384 | 96.1 | 83.7 | **70.5** | 79.0 | 46.7 |
| 512 | 101.5 | 86.1 | **84.8** | 85.1 | 52.0 |
| 1024 | — | 94.0 | **92.7** | 92.2 | 75.8 |
| 2048 | — | **130** | 149 | 162 | 103 |
| 4096 | — | **235** | 287 | 309 | 201 |

修正结论:
- **最优 persistent tile 是 kid 301(128×256),不是 303**(全 T 段 301 ≥ 303)。
- persistent 的真正甜点是 **T≈300–512**(208 已崩、kid9 未起):T=384 时 301=70us vs kid9=84us,
  **快 ~16%**;T=512/1024 只是边际(~1-2%)。
- T≤256 仍是 kid 208 最好;T≥2048 kid 9 最好(persistent 串行 M-outer 劣化)。

**新默认**(`wo_a_gemm_opus`):`T<300 → 208`,`300≤T≤1024 → 301`,`T>1024 → 9`。
全 T `err=0`,`test_batch_gemm_bshd.py` 回归通过。零拷贝 mmajor launcher 消除了 46us 转置税。
进一步追平 hipBLASLt 需 persistent 的 M-outer overlap(消除 store/prologue 串行)或 Stream-K。

---

## 追加:kid 301 深挖 —— 优化杠杆已系统穷尽

目标:让 kid 301 至少接近 hipBLASLt。ATT(att_kid301,T=1024,单 CU)stall 分解:

| 指令 | stall% |
|---|---|
| **s_barrier** | **40.2%** |
| MFMA | 17.4% |
| s_waitcnt | 14.5% |
| buffer_load | 9.8% |
| ds_read | 6.5% |
| store | 5.4% |

瓶颈仍是 s_barrier(40%)。逐个试打击它的杠杆:

| 杠杆 | 结果 |
|---|---|
| 提占用率 | kid 301 = 99KiB LDS → LDS 限制 **1 WG/CU**(VGPR 88 有富余)。但 kid 303 已是 2 WG/CU(66KiB)却不比 301 快 → **占用率不是杠杆**(同 kid 41 教训)|
| 加大 B_K=128(barrier 数减半)| **架构禁止**:校验器硬锁 `B_K == T_N·W_K/2 == 4·32/2 == 64`。要 B_K=128 需 T_N=8(→16 wave→BLOCK 1024 > 512 上限)或 W_K=64(仅有 16×16×32 MFMA)。已加注释防后人重走 |
| 单-barrier 重写 | 之前已测,更慢(barrier 与计数式部分等待耦合成细粒度流水,裸删需全量排空反而暴露 ds_read 延迟)|
| tile 变体 | 301(128×256)已是全 T 段最优;302(256×128,=hipBLASLt tile)反而最差(~124us,swizzle/wave 配置不匹配)|

**结论:kid 301 = 70us(T=384)/ 93us(T=1024)是 opus 现有 split-barrier/persistent 家族的
floor。** 那 40% barrier 是 "split-barrier 每 mma-cluster 一 barrier" + "B_K 硬锁 64 → K-loop
固定 64 次" 的结构性必然,无法在调参/削 barrier 层面消除。

## 追加:小 T 到底(kid 200 T-扫 ATT)+ s_barrier 行为解析

### kid 200(小 T 实际用的 flatmm_splitk 64×64)stall 随 T
| T (sk) | s_barrier | buffer_load | s_waitcnt | MFMA | ds_read |
|---|---|---|---|---|---|
| 64 (sk4)  | 44% | 27% | 7%  | 8% | ~0% |
| 128 (sk2) | 45% | 32% | 3%  | 9% | ~0% |
| 256 (sk0) | 47% | 25% | 18% | 5% | ~0% |

结论:kid 200 小 T 的瓶颈是 **s_barrier(44–47%）+ 内存(buffer_load/waitcnt)**,**ds_read≈0%**。
即"少用 LDS 中转"没空间——LDS 读根本不是成本(之前 46% ds_read 是 kid 208,不是现用的 200)。
44% barrier 看着高,但被 splitK 超发藏掉(T=64 追平 hipBLASLt 即证明);内存部分是该 shape 的
物理下限。**小 T 用 splitK 已摸到实际下限,收官。**

### 为什么 opus 比 hipBLASLt 多 barrier(s_barrier 行为)
`s_barrier` = **workgroup 级同步**:WG 内每个 wave 到此停住,等所有 wave 到齐再一起放行。用途是
保证 LDS 的**跨 wave 可见性**(wave 各写一部分 LDS → barrier → 才能读别的 wave 写的那部分)。它是
"全 WG 一起等",比 `s_waitcnt`(只等本 wave 自己的内存指令)重得多。

ATT 对比:hipBLASLt **waitcnt 65% / barrier 12%**;opus kid200 **barrier 44% / waitcnt 3–7%**。
opus barrier 多的三个根因:
1. **跨 wave LDS shuffle 依赖**:opus 一个 tile 由多 wave 协作,ds_write partition ≠ ds_read
   partition(你读的是别的 wave 写的),所以每次填 LDS 都得全 WG barrier;双缓冲每半 tile 一次 →
   每 K-tile ~2–4 barrier × 64 K-tile = 上百个。
2. **hipBLASLt 深预取(PGR2/PLR1)**:预取 2 tile、本地读提前 1 拍,WAR 距离大 → 每 K-step 只 1
   barrier。
3. **同步方式转移**:hipBLASLt 用 per-wave `s_waitcnt`(轻、局部)替代 `s_barrier`(重、全 WG),
   把同步压力从 barrier 转到 waitcnt(等内存,本就是内存瓶颈 shape 的物理必然)。

即:opus 是"多 wave 重度共享 LDS + 全 WG barrier 对齐",hipBLASLt 是"深缓冲拉开读写距离 +
waitcnt 主导"。要减 opus 的 barrier 就得重构成后者(PGR2/PLR1),即下节的整体重写。

## 追加:Stream-K / 深预取从零重写 —— 工作量评估

要真正追平 hipBLASLt(1.25–1.5x gap),唯一剩的路是写一个**结构不同**的新 kernel。核心要素
(照搬 hipBLASLt 的 `Cijk_..._MT256x128_..._PGR2_PLR1_..._SK3_..._WG32_8_1`):

1. **PGR2 / PLR1 深度预取**:全局预取 2 个 K-tile、本地 read 提前 1 拍。更大的 WAR 距离
   → 结构性地把每步 barrier 从 ~2-4 降到 ~1,且不靠全量排空(与我们失败的单-barrier 不同)。
   这是最大收益点,也是最难点(需重排 LDS 多缓冲 + 精细 waitcnt 计数)。
2. **4-wave / 256×128 tile + WG32_8_1 wave 布局**:填满 256 CU(1 WG/CU × 256 tile),
   避开 opus 8-wave / B_K=64 / ra-rb 耦合的全套硬锁 → 需要新的 traits + layout helper。
3. **Stream-K(SK3)**:本 shape(batch=8、tile 数≈CU 数)其实不切 K;Stream-K 是可选项,
   只在 tile < CU 的小 shape 才有用。**可先不做**,phase 1(1+2)逼近后再评估。

工作量粗估(单人):
- Phase A(新 traits + layout + 单-tile PGR2/PLR1 pipeline,非持久、非 SK):**~1.5-3 周**,
  风险高(waitcnt 计数 + LDS 多缓冲正确性是主要坑,需反复 ATT 对拍)。
- Phase B(持久化 + XCD swizzle 填 CU):**~3-5 天**(可复用现有 persistent 框架)。
- Phase C(Stream-K 动态调度 + fixup 归约):**~1-2 周**,仅在 A+B 未追平时才做。

总计逼近 hipBLASLt 约 **2-4 周**,且不保证追平 AMD 自家高度调优的库。ROI 判断:当前 kid 301
零拷贝默认已把中 T 段(模型实际 decode/短 prefill 区间)相对 kid 9 提了最多 16%,是低风险已
落地收益;Stream-K 重写属高风险大投入,建议作为独立立项、有明确性能需求时再启动。

### 尝试记录:PGR2 三缓冲原型(2026-07-07)—— 失败,第 3 次验证"粗调度必输"
在 interleave 沙盒(4-wave / B_K=32,三缓冲 ×6 LDS 放得下)实现了 **PGR2 深预取**(3-deep
ring,预取 2 K-tile,每步仅 1 个 s_barrier,vmcnt 计数保留 1 tile 在飞)。结果(err=0):

| T | PGR2 128×256(kid40)| PGR2 256×128 | PGR2 128×128 | kid301 | kid9 | hipB |
|---|---|---|---|---|---|---|
| 384 | 150 | 153 | 130 | **70** | 84 | 47 |
| 1024 | 189 | 186 | 205 | **93** | 94 | 74 |

**慢 2 倍。** 原因:只做了 PGR2(降 barrier),没做 **PLR1(细粒度 ds_read/mma 交错)**。我的
循环是"批量读 4 操作数 → lgkmcnt(0) 全排空 → 批量 4 mma",把 ds_read 延迟完全暴露,得不偿失
——和之前"单-barrier 全排空"同一个坑。**opus split-barrier 的细粒度 read/mma 交错 + 部分
lgkmcnt 计数才是它跑得快的核心**,单独降 barrier(不管 2-deep 还是 3-deep)都赢不回来。

### 尝试记录:PGR2+PLR1 三缓冲 + 公平对比(2026-07-07)—— 仍失败(第 4 次)
补上了 **PLR1**(v_a1 的 local read 提前发、用 `lgkmcnt(a_ds_read_insts)` 部分等待,与前两个
mma 重叠),凑成完整 PGR2+PLR1。并做了**公平对比**:把 kid 40 设成 8-wave 128×256 B_K=64
(与 kid 301 完全同 tile、同 wave 数,三缓冲 148KiB 放得下),只有 schedule 不同:

| T | kid40 PGR2+PLR1 三缓冲 | kid301 split-barrier 2 缓冲 | hipB |
|---|---|---|---|
| 384 | 116.7 | **70.4** | 47 |
| 1024 | 146.5 | **92.8** | 74 |
| 2048 | 250.6 | **148.4** | 103 |

**apples-to-apples 仍慢 1.5-1.7x。** 说明差距不在 4-wave/tile,而在 schedule 本身。opus 的
split-barrier 不只是"细粒度交错",还有一整套精调:分布式 waitcnt 计数 + `setprio` + 关键的
**`sched_barrier(0)` 编译器调度钉**(锁死指令序、阻止 hipcc 重排)。手写近似缺这套,一重排就崩。

**最终教训(4 次一致:单-barrier / 占用率 / PGR2 / PGR2+PLR1 全败)**:opus split-barrier 的
schedule 是 ISA 级高度调优的产物,手工在现有框架内重写半条流水**注定跑不赢**。要追平 hipBLASLt
必须完整照搬其 ISA 级 schedule(PGR2+PLR1+sched_barrier 全套 + Stream-K),这是独立的多周工程,
不是增量补丁。所有实验已回退,interleave header 恢复精确克隆,默认 kid 301 完好(70/93us)。

---

## 基线性能快照(2026-07-07,`run_perftest` 稳定计时)

opus 默认(`wo_a_gemm_opus` 自动选 kid,零拷贝)vs hipBLASLt(torch.einsum),
`N=R=1024, K=4096, batch=G=8`,全 T `err=0`:

**小 T split-K 优化后**(默认 T≤192 → kid 200 + splitK=4;192<T<300 → kid 200):

| T | opus 默认 | 选的 kid | hipBLASLt | gap | (优化前) |
|---|---|---|---|---|---|
| 64   | **21.5us** | 200/sk4 | 22.8us  | **0.94x** | (1.30x) 反超 |
| 128  | **33.2us** | 200/sk4 | 27.0us  | **1.23x** | (1.72x) |
| 192  | **43.1us** | 200/sk4 | 41.1us  | **1.05x** | 近平 |
| 256  | **49.8us** | 200     | 42.3us  | **1.18x** | (1.28x) |
| 384  | 70.5us  | 301 | 46.7us  | 1.51x | |
| 512  | 84.8us  | 301 | 52.5us  | 1.62x | |
| 1024 | 92.6us  | 301 | 75.1us  | 1.23x | |
| 2048 | 120.9us | 9   | 102.3us | 1.18x | |
| 4096 | 234.7us | 9   | 200.6us | 1.17x | |

### 小 T split-K 优化(2026-07-07)—— 真实大赢
ATT(T=64)显示 opus kid208 花 46%(barrier 27% + ds_read 19%)在 LDS 中转/同步,而 hipBLASLt
是 s_waitcnt 65% 的纯内存流式(靠 65536 WG 超发盖延迟)。**根因:小 T 输出 tile 太少填不满 256
CU。** 解法就是切 K:实测默认的 auto `splitK=0` 严重欠切,显式 **kid 200 + splitK=4** 大幅改善:
- T=64 **反超 hipBLASLt**(21.5 vs 22.8us,原 29us);
- T=128 **1.72x → 1.23x**(46→33us);T=192 近平(1.05x);T=256 用 sk0(49.8us,1.18x)。

已改进默认启发式(纯 python,不动 kernel;回归 `test_batch_gemm_bshd.py` 通过)。

**公式化 splitK(替代硬编码,batch 自适应)**:已有的 CK 路径(`batched_gemm_bf16_CK`)用
`(gfx,cu_num,B,M,N,K)→(kid,splitK)` 的 tuned CSV + `compute_batched_gemm_SplitK` 填-CU 公式。
opus wo_a 之前没接这套(opus lookup 只 key `(M,N,K)`),所以我把硬编码 `sk4` 换成 opus 版填-CU
公式 `_wo_a_auto_splitk`:`tiles = ceil(M/64)·ceil(N/64)·batch`(opus batch 在 grid.z,要乘进去),
`splitK = ceil(2·CU / tiles)`(填到 ~2 WG/CU 藏内存延迟),clamp。效果(batch=8):
T=64 sk4=0.99x、T=128 **sk2=1.14x(自动选到真最优,优于硬编码 1.23x)**、T=192 sk2=1.23x、T=256 sk0=1.20x。
- **局限(印证"batch 多变种"顾虑)**:换 batch=16 时公式在 T=128 给 sk0(1.86x,欠切)——单一填-CU
  公式无法完全泛化到所有 batch。但公式在 batch=16 给的值 ≈ 原始默认,不比原来差(对 batch=8 明显更好)。
- **真正 batch-robust 的解**:给 opus 也建 `(B,M,N,K)→(kid,splitK)` tuned CSV(像 CK 那样),
  作为后续。当前公式是 DSV4(batch=8)目标下的低风险改进。

要点:
- 优化后 **小 T(≤256)gap 收到 0.94–1.23x**(T=64 反超);过渡点 **T=384/512(1.5–1.6x)** 现在
  是最差处 —— 那里 tile 半满、切 K 已不划算但 persistent/大 tile 又填不满,正是 Stream-K 动态调度
  最占优的区间。
- 四段默认(200+sk4 / 200 / 301 / 9)覆盖 极小 / 小 / 中 / 大 T,是现有 kernel 池的逐 T 最优组合。

---

## 后续工程提案:hipBLASLt-style 新内核(多周独立立项)

> 目标:追平/接近 hipBLASLt 在 `M=1024,N=1024,K=4096,batch=8` 的 ~74us(现最优 kid 301
> = 93us,gap 1.25x)。照搬对象 = trace 里的
> `Cijk_..._MT256x128x64_MI16x16x1_GSU0_PGR2_PLR1_SK3_WG32_8_1`:
> **4-wave(WG 32×8=256 线程)/ 256×128 tile / B_K=64 / PGR2 全局预取2 / PLR1 本地读提前1 /
> Stream-K 模式3 / LDS 50KiB / 扁平 1D grid**。

### 为什么必须新写(不能打补丁)—— 已验证硬墙
- a16w16 家族硬锁:`T_M==2`;`B_K==T_N·W_K/2`(→B_K 只能 32/64);persistent `BLOCK_SIZE==512`
  (8-wave)锁死。要 4-wave+256×128+PGR2/PLR1 必须**全新 traits + 全新 layout + 全新 pipeline**。
- 4 次手改调度(单-barrier / 占用率 / PGR2 / PGR2+PLR1)全败:opus split-barrier 的
  `分布式 waitcnt + setprio + sched_barrier(0)` 是一体的,拆开重写必崩。opus 里 4-wave 缩水版
  比 8-wave 还慢(interleave 4-wave ≈120us vs kid301 8-wave 70-93us),因为 4-wave 路径没协同调。

### 设计要素(对标 hipBLASLt)
| 要素 | 内容 | 难度 |
|---|---|---|
| 新 traits | 4-wave(BLOCK=256)、256×128、B_K=64、MFMA 16×16×32、WG32×8 wave 布局 | 中 |
| 新 layout helper | lane→(m,n,k) 的 ga/sa/ra/gb/sb/rb 全部为 4-wave 重推;LDS 无 bank 冲突排布 | **高(最易错)** |
| PGR2 | 全局预取深度 2(LDS ≥3-deep 环),更大 WAR 距离 → 结构性少 barrier | 中 |
| PLR1 | 本地 read 提前 1 拍,**配 setprio/sched_barrier 精确钉住编译器指令序**(手写 4 次栽在此) | **最高** |
| Stream-K(SK3) | 扁平 1D grid、work-unit 动态切 K、尾部 fixup 归约(本 shape tile≈CU 其实不太需要) | 高(后置) |

### 分阶段(单人估,总 ~2–4 周)
- **Phase A — 4-wave 单-tile 内核(非持久/非 SK),~1.5–3 周**
  - A1:新 traits + 新 layout,先写**朴素正确**版(每步全量 barrier),打通 `err=0`。layout 推错=满屏错值。
  - A2:上 PGR2 多缓冲 + PLR1 细粒度交错,**逐指令对拍 hipBLASLt 的 code.json**,ATT 把 s_barrier
    从 40% 压到 ~10%。收益兑现点 + 最高风险点。
  - Gate:256×128 单 GEMM 追到 hipBLASLt 1.1x 以内。
- **Phase B — 持久化 + 填 CU,~3–5 天**:套现有 persistent 的 XCD swizzle 框架;256×128 → 256 tile
  正好填满 256 CU。Gate:batched 不回退。
- **Phase C — Stream-K,~1–2 周**,仅当 A+B 仍没追平才做。优先级最低(本 shape 收益有限)。

### 验证纪律(避免重蹈本轮覆辙)
- 对拍黄金参考 = hipBLASLt `code.json`,ATT 逐指令比 waitcnt/barrier/mma 排布。
- 每步 gate = `checkAllclose err=0` + **`run_perftest` 稳定计时**(不用 wall-clock)+ ATT stall 分解。
- 每 phase 独立 kid + 独立 header/tag,可回退,不碰工作默认路径。

### 风险与退出判据
- 最大风险:A2 的 PLR1 + 编译器调度钉。若 2 周内 ATT s_barrier 压不下去 / 追不到 1.2x,**止损**
  (说明手工复现 AMD 调度不现实,转评估直调 hipBLASLt 或等 opus 上游 4-wave 内核)。
- ROI 前提:需有"此 GEMM 是端到端瓶颈"的明确证据才值得投;否则 kid 301 零拷贝默认(中 T +16%)
  已是低风险落地收益。

---

## 重大发现:opus 已有 uniform+PGR2+SK 实现(gfx942 kid 10204)—— 改为移植

之前判定"opus 无 uniform 低-barrier 内核、需从零写"是**只看了 gfx950**。全 arch 搜索后发现
**gfx942 已经有目标设计**:`include/gfx942/opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh`
(kid **10204**,注释 "hipb-orientation")。特征:
- **4-wave**(BLOCK_SIZE=256, T_M=T_N=2);
- **PGR2** 深全局预取(store/refill 与 mfma 重叠);
- **waitcnt 主导 + 极少 barrier**:满屏手写 `s_waitcnt lgkmcnt(N)`,主循环区 `s_barrier` 仅
  ~2 处(vs opus gfx950 家族每 tile 2–4 个)——正是 hipBLASLt 式"每步~1 barrier";
- **uniform**:无 producer/consumer role 拆分;
- **splitK/SK** + `splitk_reduce_gfx942`。

参考物性质:和 mono_tile 当年"从 yk_gcn 移植"同套路 —— 有可跑、可对拍的基线,不是凭空推 layout。
**工作量从"多周从零"降到"移植 + 调 MFMA/LDS,数天级"。**

### 移植目标 / 需对齐的点(gfx942 → gfx950)
| 项 | gfx942(现有 kid 10204)| gfx950 目标 | 备注 |
|---|---|---|---|
| MFMA | 16×16×**16** BF16 | 16×16×**32** BF16 | K 维翻倍;`opus_gemm_asm_mma16x16x16` → gfx950 的 32-K MMA |
| 每 MMA 的 K | W_K=16 | W_K=32 | E_K = B_K/W_K 随之变 |
| layout helper | 按 16×16×16 lane→(m,n,k) | **按 16×16×32 重推** ra/rb/sa/sb | 唯一较重的改动;拿 gfx942 版逐步改 |
| tile | 96×128×{64,128} | 起步 96×128 或直接对齐 hipb 的 256×128 / 256×256 | 先小后大 |
| LDS 预算 | ≤64KiB(LDS_DEPTH=1 单缓冲)| ≤160KiB(可上 PGR2 更深)| gfx950 更宽松 |
| reduce | `splitk_reduce_gfx942` | `splitk_reduce_gfx950`(已存在)| 直接换 |
| 调度骨架 | PGR2 / SK / waitcnt / barrier 排布 | **原样复用** | 核心价值,不重写 |
| 校验器/codegen | gfx942 emit | gfx950 新 tag + 校验(参考 interleave 接线)| 常规接线 |

### 建议流程
1. **(用户在 gfx942 上先做)** 跑通 kid 10204,确认它 uniform+PGR2+低barrier,记录它在
   `1024×1024×4096×8`(或 MI300 对应 shape)的 perf + ATT stall 分解,作为移植基线。
2. gfx950 新 tag(如 `a16w16_uniform_pgr2`),复制 gfx942 kernel body,换 MMA 16×16×32 + 重推
   layout + 换 reduce,先 `err=0`。
3. ATT 对齐调度(barrier/waitcnt 密度对齐 gfx942 参考),再逐 tile 档 + 填 CU,测 vs kid 301/hipB。

### 其它可参考的 gfx942 现代 pipeline
`kbuf2v`(K 双缓冲)、`kbuf2v_bk128`、`wave_k_coop`(wave 间 K 协作)、`quad_mfma32_kbuf1`。

---

## 附:复现命令

```bash
# 容器内,PYTHONPATH 指向本 checkout
export PYTHONPATH=/home/jun_chen2_qle/yzhou_aiter/aiter
# 正确性 + 回归
python op_tests/test_batch_gemm_bshd.py -k 0 8
python op_tests/test_batch_gemm_bhsd.py  -k 0 8
# 强制重编译 opus 模块(改了 csrc 后)
rm -f  aiter/jit/module_deepgemm_opus.so
rm -rf aiter/jit/build/module_deepgemm_opus
# ATT(需先装 trace-decoder,见 §0):rocprofv3 -i <att.yaml> -- python <driver>.py
```
