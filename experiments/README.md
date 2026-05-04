# Phase 0 Risk Validation Experiments

Two targeted experiments to gate the HRNVS pipeline design decisions before
committing to full implementation.

---

## 目录结构

```
experiments/
├── configs.py                  # 场景列表、分辨率、`priors/` 常量、阈值
├── requirements.txt
│
├── models/
│   ├── __init__.py
│   └── hr_head.py              # Phase 2.2 HR Geometric Prior Head
│
├── utils/
│   ├── colmap_reader.py        # COLMAP binary 读取 + 稀疏深度计算
│   ├── dataset.py              # MipNeRF360 帧加载 + COLMAP 深度插值
│   ├── metrics.py              # PSNR/SSIM/AbsRel/ScaleInvL1 + Sobel 边缘
│   ├── warp.py                 # 反向投影 warp (backward warping)
│   └── swinir_wrapper.py       # SwinIR ×4 SR（首次运行自动下载）
│
├── task01_2dsr_consistency.py  # Task 0.1 主脚本
├── task02_vggt_geometry.py     # Task 0.2 主脚本
├── task22_hr_head_smoke.py   # Phase 2.2 HR Head 形状/参数量自检
├── task02_oracle_train.sh      # 生成 oracle 深度（mip-splatting 训练）
├── task02_oracle_render.py     # 深度图转换为 .npy
│
├── run_task01.sh               # Task 0.1 快速启动
└── run_task02.sh               # Task 0.2 快速启动（包含 oracle 生成）
```

---

## 环境安装

```bash
cd experiments
pip install -r requirements.txt

# mip-splatting 依赖（Task 0.2 oracle 需要）
cd ../mip-splatting
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

---

## 数据准备

在 `configs.py` 中修改：

```python
MIPNERF360_ROOT = "/your/path/to/mipnerf360"
```

期望目录结构：

```
mipnerf360/
  garden/
    images/          # 原始全分辨率
    images_2/        # 1/2 分辨率（oracle 训练用）
    images_4/
    images_8/        # ← LR 源 (Task 0.1 & 0.2)
    priors/          # ← StableSR 等 HR cache（与 images_8 同级，帧 stem 对齐）
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
  kitchen/
  bonsai/
  room/
  counter/
```

---

## Task 0.1 – 2DSR 视角不一致严重性测试

**原理**：对每个场景取 8 帧 → 读取 StableSR **预生成** SR（`scene/priors/` 或通过 `--sr_dir`）→ 用 COLMAP GT 相机做跨视角 backward warp → 比较 warp 结果与直接 SR 结果的 PSNR/SSIM。**也可**不传 `--sr_dir` 时使用内置 SwinIR 实时推理（见脚本说明）。

```bash
bash run_task01.sh /path/to/mipnerf360 ./results/task01 cuda
# 或直接调用 Python：
python task01_2dsr_consistency.py \
    --data_root /path/to/mipnerf360 \
    --output_dir ./results/task01 \
    --device cuda \
    --save_visuals
```

**预计耗时**：~30 分钟（5 场景，8 帧，A100）

**判定阈值**：

| PSNR(full) | 结论 | 后续动作 |
|---|---|---|
| ≥ 28 dB | 🟢 不一致可忽略 | Narrative 简化 |
| 22–28 dB | 🟡 中度不一致 | Confidence weighting 即可 |
| < 22 dB | 🔴 严重不一致 | 必须实现 View-Consistent SR 模块 |

**输出**：
- `results/task01/scene_summary.csv` — 每场景 PSNR/SSIM（全局 + Sobel 边缘）
- `results/task01/summary.json` — 汇总 + 决策结论
- `results/task01/<scene>_pairs.csv` — 每视角对详细指标
- `results/task01/visuals/` — [ref SR | warped SR | ×5 diff] 对比图（`--save_visuals`）

---

## Task 0.2 – VGGT 在 200×200 输入下的几何 Fidelity

**原理**：Frozen VGGT 在 200×200 LR 输入下预测深度 → 与 vanilla Mip-Splatting（HR 全分辨率训练 30k iter）渲染的 oracle 深度比较

### Step 1：生成 oracle 深度（~30 分钟/场景）

```bash
bash task02_oracle_train.sh /path/to/mipnerf360 ./results/task02/oracle 0
```

### Step 2：VGGT 推理 + 比较

```bash
python task02_vggt_geometry.py \
    --data_root   /path/to/mipnerf360 \
    --oracle_root ./results/task02/oracle \
    --output_dir  ./results/task02 \
    --device cuda
```

**或一键运行两步**：

```bash
bash run_task02.sh /path/to/mipnerf360 ./results/task02/oracle ./results/task02 cuda 0
```

**判定阈值**：

| AbsRel | 结论 | 后续动作 |
|---|---|---|
| < 0.10 | 🟢 VGGT 直接可用 | 直接进入 Phase 2 HR Head 训练 |
| 0.10–0.20 | 🟡 需 HR Head 强化 | 加强深度监督 / dual-branch HR Head |
| ≥ 0.20 | 🔴 需 LoRA 微调 | 用 oracle 深度监督 LoRA fine-tune VGGT |

**输出**：
- `results/task02/scene_summary.csv`
- `results/task02/summary.json`
- `results/task02/<scene>_frames.csv`

---

## Phase 2.2 – HR Geometric Prior Head（结构）

模块：`models/hr_head.py` 中 `HRGeometricPriorHead`。条件输入为 **LR 尺度** 拼成的张量：VGGT 深度（可选 log）、LR RGB、以及 **StableSR `priors/` 的 HR 图经双线性下采样到 LR** 作为第三路条件；输出 `depth_hr` / `normal_hr` / `confidence_hr`（默认 800×800）。

自检：

```bash
cd experiments
python task22_hr_head_smoke.py --device cuda
```

数据集加载：`utils/dataset.load_scene_frames(..., prior_subdir="priors")` 会为存在文件的帧附加 `prior_sr_hr`。

---

## 注意事项

1. **StableSR `priors/`**：每张 SR 图与 `images_8` 下同 stem；`configs.PRIORS_SUBDIR` 默认 `priors`。
2. **SwinIR 权重**：Task 0.1 **未指定 `--sr_dir` 且走 SwinIR 路径时**会自动从 GitHub releases 下载
   (~130 MB)，存放在 `third_party/weights/`。

3. **VGGT 权重**：Task 0.2 首次运行时自动从 HuggingFace 下载
   (~4 GB)，存放在 `../vggt/model.pt`。

4. **COLMAP 稀疏深度质量**：`images_8` 下的特征点较少，深度插值可能
   在部分场景存在大面积空洞。可通过 `--image_subdir images_4` 改用更
   高分辨率的 LR 来增加稀疏点密度。

5. **Oracle 深度格式**：mip-splatting 默认不输出深度图；
   `task02_oracle_render.py` 会尝试从渲染输出中提取深度通道（EXR/16-bit PNG）。
   如果深度渲染不可用，可修改 `mip-splatting/render.py` 添加深度输出。
