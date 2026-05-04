# HRNVS 项目最终版任务清单

> 综合所有讨论后的项目章程。本文档将作为后续工作的统一参考。

---

## 一、项目定位

| 项 | 内容 |
|---|---|
| **任务** | HRNVS (High-Resolution Novel View Synthesis) |
| **Setting** | 200×200 LR multi-view → 800×800 HR rendering（4× SR，标准 HRNVS） |
| **目标 1（渲染）** | PSNR / SSIM / LPIPS 超过 Mip-Splatting + 2DSR、SRGS / GaussianSR / SuperGS 等 per-scene 3DSR SOTA |
| **目标 2（几何）** | Chamfer / F-score / Normal Consistency 接近 GOF / 2DGS 级别（在有 GT mesh 的数据集上） |
| **范式** | Feed-forward HR geometric prior + per-scene 3DGS optimization（混合范式，非纯 feed-forward） |

---

## 二、核心 Contributions（论文卖点）

1. **Geometry-Anchored Initialization & Supervision**：用 HR multi-view consistent 几何 prior 取代 SfM 初始化、并在 per-scene 优化全程作 confidence-weighted supervision，锁住几何使其不被 2DSR 视角不一致拉扯。
2. **Geometry-Aware Densification** ⭐ 核心 novelty：把 prior 几何信息注入 GS 的 split / clone / prune 决策与方向，使高斯沿 prior 表面生长。
3. **Mip-Aware Scale Bootstrapping**：从 HR depth 推算每像素几何尺度，bootstrap Mip-Splatting 的 3D / 2D filter 参数。
4. **HR Geometric Prior Module (building block)**：基于 frozen VGGT + 自训 HR Head，在 LR 输入下输出 HR multi-view consistent depth / normal / confidence；HD-VGGT 开源后可热替换。

---

## 三、整体 Pipeline

```text
┌── Stage 1: Feed-Forward (一次推理 < 1s) ──────────────────────┐
│                                                                │
│  LR multi-view (200×200)                                       │
│         │                                                      │
│         ├──► [Frozen VGGT] ──► coarse depth + cameras          │
│         │                                                      │
│         ├──► [HR Head (trained)] ──► D_HR, N_HR, C_HR (800×800)│
│         │                                                      │
│         └──► [SwinIR (frozen)] ──► I_2dsr (800×800)            │
│                                  │                             │
│                  [VGGT cameras + epipolar attn]                │
│                                  │                             │
│                                  ▼                             │
│                          F_VCSR (view-consistent SR feats)     │
└────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌── Stage 2: Per-Scene Optimization (5–15 min) ─────────────────┐
│                                                                │
│  Init = unproject(D_HR)                                        │
│                                                                │
│  Loss = L_rgb_VCSR + L_d·C_HR + L_n·C_HR + L_b                 │
│       + L_OF + L_flat + L_mip                                  │
│                                                                │
│  Densify = Geometry-Aware                                      │
│  (split/prune by prior surface, split direction by N_HR)       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                   HR 3DGS (rendering @ any res)
```

---

## 四、数据集与评测协议

### 数据集（Mip-Splatting 标准）
- **Mip-NeRF360** outdoor: bicycle, flowers, garden, stump, treehill
- **Mip-NeRF360** indoor: room, counter, kitchen, bonsai
- **Tanks-and-Temples**: truck, train（提供几何评测 GT）
- **Deep Blending**: drjohnson, playroom

### SR Setting
- 训练 LR：`images_8`
- 测试 HR：`images_2` / `images_4` / `full`

### 指标
- **渲染**：PSNR / SSIM / LPIPS（用 `mip-splatting/lpipsPyTorch`）
- **几何**（仅 T&T / DTU 子集）：Chamfer / F-score / Normal Consistency
- **多分辨率**：1× / 2× / 4× rendering PSNR
- **效率**：per-scene optimization wall-clock time

---

## 五、与现有工作差异化

| 维度 | SR3R | SRGS / GaussianSR / SuperGS | HD-VGGT | **本工作** |
|---|---|---|---|---|
| 范式 | Pure feed-forward | Per-scene + 2DSR pseudo | Feed-forward 几何 | **Hybrid: FF prior + per-scene** |
| Setting | 64→256 sparse | 200→800 dense | 通用 reconstruction | **200→800 dense (标准 HRNVS)** |
| 几何 prior | 无 | 无 | 自身就是 | **复用 + 注入到 GS** |
| 2DSR 监督 | 反对 | 直接用 | N/A | **VC-SR features 后用，confidence-weighted** |
| 几何评测 | 不报 | 不报 | 报 | **作为核心卖点报** |

---

## 六、已知风险与状态

| 风险 | 严重度 | 状态 | 缓解 |
|---|---|---|---|
| 1. VGGT 在 200×200 LR 几何 fidelity | 🟡 中 | 待 Phase 0.2 验证 | LoRA 微调 / HR Head 强化 |
| 2. 2DSR 视角不一致是 PSNR 天花板 | 🔴 致命 | 待 Phase 0.1 验证 | View-Consistent SR features 模块 |
| 3. HR Head 训练 GT depth 来源 | 🟡 中 | Mip-NeRF360 dense view + oracle GS distill | – |
| 4. HD-VGGT 不直接适用 SR setting | 🟢 缓解 | 用 HR GT 作训练 guidance | – |
| 5. 与 AnySplat / FLARE 差异化 | 🟡 中 | narrative 重心放 Stage 2 | – |
| 6. 几何指标只能小数据集报 | 🟡 中 | T&T + DTU zero-shot | – |

---

## 七、分阶段任务清单

### Phase 0：风险排雷（1–3 天）⚠️ 优先级最高

- [x] **0.3** 查 SRGS / GaussianSR 在 Mip-NeRF360 4× SR 下的真实 PSNR / SSIM 数字 *(用户已 done，待提供数字给协作方 anchor)*
- [ ] **0.1** 2DSR 视角不一致严重性测试（半天）
  - 数据：Mip-NeRF360 5 场景（garden, kitchen, bonsai, room, counter）
  - 流程：images_8 → SwinIR → 跨视角 warp → edge-weighted PSNR / SSIM
  - 阈值：PSNR > 28 = 可忽略；22–28 = 中度（confidence weighting 即可）；< 22 = 严重（必做 VC-SR module）
- [ ] **0.2** VGGT 在  LR 几何 fidelity 测试（1 天）
  - 数据：同上 5 场景
  - 流程：frozen VGGT → depth；vanilla GOF (HR full, 30k iter) → oracle depth；对比 scale-invariant L1, AbsRel
  - 阈值：AbsRel < 0.10 直接可用；0.10–0.20 需 HR Head 强化；> 0.20 需 LoRA fine-tune VGGT
- [x] **0.4** 输出 1 页 risk memo，决策矩阵触发后续路径（见 `experiments/RISK_MEMO_PHASE0.md`）

### Phase 1：基础设施（1 周）

- [ ] **1.1** 选定 codebase 骨架：fork 本地 `mip-splatting/`（含 GOF / 2DGS 模块）作为 base
- [ ] **1.2** 扫一眼 `mip-splatting/hybrid_sdfgs/`，看是否有可复用的 SDF+GS 混合代码
- [ ] **1.3** Mip-NeRF360 / T&T / Deep Blending 数据下载 + LR/HR 配对生成 pipeline
- [ ] **1.4** Baseline 复现：vanilla Mip-Splatting + 2DSR / SRGS（取较强者）@ Mip-NeRF360
- [ ] **1.5** 跑 vanilla GOF + HR GT 监督，得到几何上限参考

### Phase 2：Stage 1 模块实现（1.5 周）

- [ ] **2.1** Frozen VGGT 推理 wrapper（输入 LR multi-view → 输出 depth + cameras + features）
- [ ] **2.2** HR Head 实现（dual-branch，借鉴 HD-VGGT 思路；约 30M 参数）
- [ ] **2.3** HR Head 训练（数据：Mip-NeRF360 dense view + oracle GS distillation；A100 单卡 1–2 天）
- [ ] **2.4** View-Consistent SR features 模块（复用 VGGT cameras + epipolar attention）
- [ ] **2.5** Stage 1 端到端验证：50 场景 AbsRel + cross-view consistency 指标

### Phase 3：Stage 2 模块实现（1.5 周）

- [ ] **3.1** 几何 prior 注入：unproject init + L_d/L_n/L_b confidence-weighted supervision
- [ ] **3.2** ⭐ **Geometry-Aware Densification 实现**（核心 novelty）
  - split criterion：prior surface deviation + grad
  - split direction：along N_HR tangent plane
  - prune criterion：远离 prior surface 的孤立 Gaussian
- [ ] **3.3** Mip-aware scale bootstrap：从 D_HR 推 ε_3d / δ
- [ ] **3.4** View-Consistent SR loss 集成（替代原始 2DSR 监督）
- [ ] **3.5** 5 场景 overfit 测试 + 调梯度爆炸 / NaN / loss schedule

### Phase 4：实验与调优（2–3 周）

- [ ] **4.1** 主对比表：Mip-NeRF360 + T&T + Deep Blending 完整测试集（vs Mip-Splatting+SR / SRGS / GaussianSR / SuperGS / GOF+SR）
- [ ] **4.2** 几何评测：T&T (truck, train) + DTU 子集（Chamfer / F-score / Normal Consistency）
- [ ] **4.3** Ablation 表：去掉 HR Head / Geo-Aware Densify / VC-SR / Mip-aware 各自的影响
- [ ] **4.4** 多分辨率渲染表：1× / 2× / 4× 各分辨率 PSNR
- [ ] **4.5** 效率分析：vs SRGS 的 wall-clock time
- [ ] **4.6** 跨数据集零样本：HR Head 在 Mip-NeRF360 训，零样本到 ScanNet / DTU

### Phase 5：论文写作（1.5–2 周）

- [ ] **5.1** Method section（架构图、伪代码、loss 设计）
- [ ] **5.2** Experiments section（4 个实验表 + 定性比较）
- [ ] **5.3** Intro / Related Work / Abstract
- [ ] **5.4** Polish + supplementary + 视频 demo

---

## 八、时间表与里程碑

| Phase | 时间 | 决策点 |
|---|---|---|
| 0. 风险排雷 | 1–3 天 | 路线 GO / FIX / REPLAN |
| 1. 基础设施 | 1 周 | baseline 复现 |
| 2. Stage 1 | 1.5 周 | HR Head 几何质量过线 |
| 3. Stage 2 | 1.5 周 | 端到端 5 场景 PSNR 过 SRGS |
| 4. 实验 | 2–3 周 | 全实验数字到位 |
| 5. 论文 | 1.5–2 周 | 投稿 |
| **合计** | **~9–12 周** | |

---

## 九、立即下一步（今天 / 明天）

1. **【5 分钟，待用户】** 把 Phase 0.3 查到的 SRGS / GaussianSR PSNR / SSIM 数字告诉我，作为目标线
2. **【今天/明天，半天】** 任务 0.1：跑 SwinIR + 跨视角 warp 实验
3. **【明天，1 天】** 任务 0.2：VGGT geometry fidelity 实验
4. **【可选，30 分钟】** 我顺手扫 `mip-splatting/hybrid_sdfgs/` 看是否有可复用代码

---

## 十、待用户确认 / 提供

- [ ] **目标会议 / 截稿日期**（决定 Phase 5 起始时间）
- [ ] **可用 GPU 资源**（决定 Phase 4 并行度）
- [ ] **SRGS / GaussianSR 的实测 PSNR 数字**（anchor 目标线）
- [ ] **是否需要把这份清单存为 `PROJECT_PLAN.md` 文件**？

---

需要我把这份清单写到一个 `PROJECT_PLAN.md` 文件吗？或者你想先调整哪一部分（比如某个 Phase 的拆解还需要更细，或者某个 task 的优先级要调）？