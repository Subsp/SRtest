# Phase 0.4 Risk Memo（一页）

**项目**：HRNVS（200×LR → 800×HR，4× SR）  
**日期**：2026-05-04  
**结论**：GO —— 默认进入 Phase 1 基础设施 + Phase 2.2 HR Head  

---

## 1. Risk 矩阵与决策（写死）

| ID | 风险 | Phase 0 证据 | 严重度 → 处置 | Narrative / 工程 |
|----|------|--------------|-----------------|-------------------|
| R1 | 2DSR 跨视角不一致拖 PSNR | 独立 warp→PSNR 路径受深度/周期性纹理干扰，**未作为最终量化指标**；**prior 相对无 prior 确有增益（团队已确认）** | 🟡 中 | **Confidence weighting**（来自 HR Head 的 C）压住 2DSR 监督；**不对外 claim「已严格量化 SOFSR 视角不一致」**，改表述为「经验上 prior 有效 + 需 view-aware 损失设计」 |
| R2 | VGGT@200×LR 几何不可用 | **Frozen VGGT** vs **GS-oracle 深度代理**（训练 checkpoint depth-as-color render）：**AbsRel ≈ 0.087 &lt; 0.10**（7/8 帧有 oracle，1 帧为 train/test 对齐缺失） | 🟢 低 | **默认 frozen VGGT**；暂不优先 LoRA VGGT。**HR Head** 承担 refine + confidence |

---

## 2. Caveats（审稿 / 对内必须写清）

1. **Oracle 深度**：非独立 COLMAP/SfM 真值；为「已收敛 GS + rasterization」下的 **proxy upper-bound**。结论句式：**「若 VGGT 连 proxy 都难对齐则更差；能对齐则说明 prior 链路里 VGGT 非明显短板」**。
2. **样本**：kitchen 单场景 smoke；**VGGT 统计为 8 帧中 7 帧**（oracle 文件名与抽样一时未全覆盖时可复现）。
3. **SOFSR / SOF**：当前仓库与训练资源以 **LR 设定下的 SOF 实验**为主；本 memo **不评价完整 SOFSR 论文级机制**（mesh fusion、extension 等），仅锁定 **VGGT + prior 路线** 的 go/no-go。

---

## 3. GO / NO-GO

| 决策 | 条件 |
|------|------|
| **GO（默认）** | R2 阈值满足 **AbsRel &lt; 0.10**；R1 **prior 有增益**。**→ Phase 1 + Phase 2.2（HR Head）并行准备。** |
| **NO-GO / 回调** | 多场景复核后 **AbsRel 系统性 ≥ 0.20**：优先 **LoRA / SR-aware fine-tune VGGT**，暂缓纯 frozen 叙事。 |

**阈值回顾（task.md）**  

- VGGT：**&lt; 0.10** 直接可用；**0.10–0.20** HR Head 强化；**≥ 0.20** LoRA。  
- 2DSR 不一致（原 warp/LPIPS 预案）：本会话未采用为最终数字；工程上对应 **confidence weighting**。

---

## 4. 下一阶段（默认路径）

1. **Phase 1**：数据协议固定（`images_8` / `images_2`）、baseline 表（Mip-Splatting LR + 能接受的对照）、评测脚本对齐。  
2. **Phase 2.2**：**HR Head**（VGGT frozen 特征 / cameras + HR depth·normal·confidence），与 Stage 2 的 **confidence-weighted** 几何监督衔接。  
3. **备忘**：补齐多场景 VGGT AbsRel；固定抽样 seed / 帧白名单消除 7/8 帧问题。

---

*本文件对应 task.md Phase **0.4**；Phase 0.1 / 0.2 的结论以上表为准，后续 narrative 不得与此矛盾。*
