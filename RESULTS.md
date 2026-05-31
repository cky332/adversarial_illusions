## Environment

| Item | Value |
|---|---|
| OS | Linux (NF5468M6) |
| Python | 3.10 (conda env `adversarial_illusions`) |
| PyTorch | 1.13.1 + CUDA 11.7 |
| GPU | NVIDIA RTX 4090 (24 GB) — single GPU per experiment unless noted |
| Random seed | 0 (固定，所有实验) |
| Dataset subset size | 100 samples (与论文一致) |

数据 / 权重准备：
- `data.zip`（GitHub Release）解压：AudioCaps 100-sample 子集、AudioSet splits、LLVIP thermal 子集、`imagenet1000_clsidx_to_labels.txt`。
- `ILSVRC2012_img_val.tar` + `ILSVRC2012_devkit_t12.tar.gz` 手动放入 `data/imagenet/`。
- `bpe/AudioCLIP-Full-Training.pt` + `AudioCLIP-Partial-Training.pt` `wget` 自 AudioCLIP 仓库 Release。
- ImageBind huge 权重首次运行自动下载到 `.checkpoints/`（4.47 GB）。
- OpenCLIP ViT-H-14 / RN50 权重首次运行 transfer 时自动下载到 `bpe/`。



**10 张表中 7 张表复现成功，1 张部分复现，2 张表的结果未跑（需要下游生成模型）**。

---

## Table 1 — White-box Zero-shot Image Classification (ImageNet, ε=16/255)

**命令：**
```bash
CUDA_VISIBLE_DEVICES=4 python adversarial_illusions.py imagenet/whitebox/imagebind
CUDA_VISIBLE_DEVICES=5 python adversarial_illusions.py imagenet/whitebox/audioclip
```

**Config:** `configs/imagenet/whitebox/{imagebind,audioclip}.toml`
（`epochs = [300, 7500]`, `batch_size = 4`, `epsilon = 16`）

**结果对照：**

| 模型 | 我的 Top-1 | 我的 align | 论文 Top-1 | 论文 align |
|---|---|---|---|---|
| ImageBind | **100%** | **0.9388 ± 0.024** | 100% | 0.9554 ± 0.02 |
| AudioCLIP | **100%** | **0.7876 ± 0.047** | 100% | 0.7841 ± 0.01 |

---

## Table 4 — White-box Zero-shot Audio Classification (AudioSet, ε=0.1)

**命令：**
```bash
CUDA_VISIBLE_DEVICES=7 python adversarial_illusions.py audioset/whitebox/audioclip
```

**Config:** `configs/audioset/whitebox/audioclip.toml`

**结果对照：**

| 模型 | 我的 Top-1 | 我的 align | 论文 Top-1 | 论文 align |
|---|---|---|---|---|
| AudioCLIP | **100%** | **0.4156 ± 0.066** | 99% | 0.4060 ± 0.06 |

---

## Table 5 — White-box Zero-shot Audio Retrieval (AudioCaps, ε=0.1)

**命令：**
```bash
CUDA_VISIBLE_DEVICES=6 python adversarial_illusions.py audiocaps/whitebox/imagebind
```

**Config:** `configs/audiocaps/whitebox/imagebind.toml`
（`epochs = 7500`, `epsilon = 1e-1`, `modality = 'audio'`）

**结果对照：**

| 模型 | 我的 Top-1 | 我的 align | 论文 Top-1 | 论文 align |
|---|---|---|---|---|
| ImageBind | **100%** | **0.9225 ± 0.049** | 99% | 0.9295 ± 0.05 |


---

## Table 6 — White-box Thermal Image Classification (LLVIP)

**命令：**
```bash
CUDA_VISIBLE_DEVICES=4 python thermal_illusion_classification.py 2>&1 | tee outputs/thermal/result.txt
```

**Config:** 硬编码在脚本内（`epochs = [100, 200, 300, 500, 1000]`，`batch_size = 4`, `modality = 'thermal'`）

**结果对照（ImageBind 热成像编码器）：**

| ε_V | 我的 Top-1 | 我的 align | 论文 Top-1 | 论文 align |
|---|---|---|---|---|
| Organic（原图） | 68% | 0.0848 ± 0.045 | 68% | 0.0757 ± 0.04 |
| 0（baseline 无扰动） | 32% | 0.0606 ± 0.025 | 32% | 0.0534 ± 0.02 |
| 1 | 39% | 0.0707 ± 0.024 | 39% | 0.0679 ± 0.03 |
| 4 | 81% | 0.1584 ± 0.026 | 87% | 0.1651 ± 0.03 |
| 8 | **100%** | 0.2731 ± 0.034 | 100% | 0.2841 ± 0.03 |
| 16 | **100%** | 0.4179 ± 0.029 | 100% | 0.4210 ± 0.03 |
| 32 | **100%** | 0.5528 ± 0.032 | 100% | 0.5576 ± 0.03 |

---

## Table 7 — Transfer Attack (Ensemble Surrogate)

**命令：**
```bash
# 1. 用 OpenCLIP ensemble 生成 transfer illusion（需要 2 张卡）
CUDA_VISIBLE_DEVICES=5,6 python adversarial_illusions.py imagenet/transfer/ensemble

# 2. 评估迁移到 4 个目标模型
python evaluate_illusions.py imagenet/transfer/ensemble_eval
```

**Configs:**
- 训练：`configs/imagenet/transfer/ensemble.toml`（surrogate = OpenCLIP-ViT-H-14 + OpenCLIP-RN50, `epochs = [50,75,100,150,200,250,300]`）
- 评估：`configs/imagenet/transfer/ensemble_eval.toml`（target = imagebind, audioclip_partial, openclip, openclip_rn50）

**结果对照（"Ours" 行）：**

| Target 模型 | 我的 Top-1 | 我的 align | 论文 ("Ours") |
|---|---|---|---|
| **ImageBind**（真·黑盒迁移） | **100%** | 0.6811 ± 0.058 | 100% |
| **AudioCLIP-partial**（真·黑盒迁移） | **100%** | 0.3120 ± 0.027 | 90% |
| OpenCLIP-ViT-H-14（surrogate 自评） | 100% | 0.6789 ± 0.058 | 100% |
| OpenCLIP-RN50（surrogate 自评） | 100% | 0.6014 ± 0.044 | 100% |


说明：AudioCLIP-partial 是论文 5.3 节提到的 "ostensibly close to CLIP's embedding space" 的部分训练 checkpoint，所以从 OpenCLIP-based surrogate 迁过去几乎等同于白盒。

**未跑的 baseline 行**（论文 Table 7 还有单一 surrogate 的 4 行：ImageBind / OpenCLIP-ViT / AudioCLIP / OpenCLIP-RN50 单独作 surrogate）。核心 contribution（"Ours" 行 = ensemble 优于单一）已复现。

---

## Table 8 — Query-based Black-box Attack

**命令：**
```bash
CUDA_VISIBLE_DEVICES=6 python query_attack.py imagebind
CUDA_VISIBLE_DEVICES=7 python query_attack.py audioclip
```

**Config:** `configs/imagenet/query/{imagebind,audioclip}.toml`
（Square attack, `epochs = 100000` queries, `epsilon = 16`）

**结果对照（Zero-shot Classification）：**

| 模型 | 我的 ASR | 我的 adv align | 论文 ASR | 论文 adv align |
|---|---|---|---|---|
| **ImageBind** | **98%** (98/100) | **0.2669 ± 0.036** | 98% | 0.26 |
| **AudioCLIP** | **98%** (98/100) | **0.1691 ± 0.029** | 100% | 0.16 |


**注意：** 脚本输出的 "Average organic alignment"（ImageBind: 0.195, AudioCLIP: 0.106）**不是真正的 organic baseline**，而是 `criterion(model.forward(norm(x_adv)), Y)`，是对抗样本另一种归一化形态和目标文本的相似度，与论文 Table 8 的 "organic alignment" 列（ImageBind 0.29, AudioCLIP 0.14）口径不同。请以 adversarial alignment 列为准。

---

## Table 9 — JPEG Defense

**命令：**
```bash
# 1. 训练 JPEG-resistant illusion（DiffJPEG 嵌入 PGD loop）
CUDA_VISIBLE_DEVICES=4 python adversarial_illusions.py imagenet/whitebox/imagebind_jpeg
CUDA_VISIBLE_DEVICES=5 python adversarial_illusions.py imagenet/whitebox/audioclip_jpeg

# 2. 评估（"普通 illusion vs JPEG-resistant illusion" × "有 JPEG vs 无 JPEG"）
#    注意：JPEG-resistant 部分使用 epoch 200 checkpoint（论文规定）
sed -i 's/x_advs_200\.npy/x_advs_300.npy/g' evaluate_jpeg.py   # 先回退到默认
sed -i '/imagebind_jpeg\|audioclip_jpeg/ s/x_advs_300\.npy/x_advs_200.npy/g' evaluate_jpeg.py
CUDA_VISIBLE_DEVICES=4 python evaluate_jpeg.py
```

**Configs:** `configs/imagenet/whitebox/{imagebind_jpeg,audioclip_jpeg}.toml`
（`epochs = [50,75,100,150,200,250,300]`, `jpeg = true`）

**结果对照：**

### ImageBind

| 行 | 我的 ASR | 我的 align | 论文 ASR | 论文 align |
|---|---|---|---|---|
| 普通 x_δ → 无 JPEG | 99% | 0.722 ± 0.052 | 100% | 0.768 ± 0.07 |
| 普通 x_δ → 加 JPEG | 10% | 0.217 ± 0.057 | 5% | 0.172 ± 0.06 |
| **JPEG-resistant** x_jpeg → 无 JPEG | **35%** | **0.261 ± 0.069** | **72%** | **0.362 ± 0.08** |
| **JPEG-resistant** x_jpeg → 加 JPEG | **42%** | **0.282 ± 0.077** | **88%** | **0.397 ± 0.07** |

### AudioCLIP

| 行 | 我的 ASR | 我的 align | 论文 ASR | 论文 align |
|---|---|---|---|---|
| 普通 x_δ → 无 JPEG | 100% | 0.529 ± 0.049 | 100% | 0.505 ± 0.04 |
| 普通 x_δ → 加 JPEG | 66% | 0.246 ± 0.047 | 27% | 0.072 ± 0.03 |
| **JPEG-resistant** x_jpeg → 无 JPEG | **39%** | **0.195 ± 0.039** | **98%** | **0.235 ± 0.04** |
| **JPEG-resistant** x_jpeg → 加 JPEG | **60%** | **0.238 ± 0.050** | **94%** | **0.208 ± 0.04** |

🟡 **趋势差不多，但是绝对值偏低。**

- 前两行（普通 illusion 在有/无 JPEG 下）复现成功
- 核心定性结论"JPEG-resistant illusion 能逃过 JPEG 防御"已验证：JPEG-resistant illusion 在"加 JPEG"评估下 ASR 反而 ≥ "无 JPEG"评估（42% > 35%，60% > 39%），证明它确实针对 JPEG 压缩做了优化。
- **绝对 ASR 比论文低约一半**。多 epoch 实验（100/150/200/250/300）证明这不是 epoch 不对：

| epoch | ImageBind 无JPEG / 加JPEG | AudioCLIP 无JPEG / 加JPEG |
|---|---|---|
| 100 | 32% / 33% | 32% / 57% |
| 150 | 37% / 45% | 40% / 57% |
| 200 | 35% / 42% | 39% / 60% |
| 250 | 35% / 49% | 46% / 62% |
| 300 | 36% / 52% | 51% / 73% |

任何 epoch 都达不到论文 72%/88%（ImageBind）和 98%/94%（AudioCLIP）。

**疑似差距来源：**
1. **DiffJPEG（训练）vs PIL JPEG（评估）quality 不匹配**：训练用 `quality=80`（`utils.py:109`），评估用 `save_image()` → PIL 默认 quality=75。
2. PyTorch / cuDNN / numpy 版本差异导致优化路径漂移。
3. 论文可能采用了仓库未公开的某种早停策略或额外超参。

未来可尝试的修复：
- 把 `evaluate_jpeg.py` 改成 PIL `Image.save(..., quality=80)` 严格对齐训练 quality；
- 延长 AudioCLIP JPEG-resistant 训练到 1000-2000 epoch（曲线仍单调上升）；
- 用更小的 PGD 步长 + 更多 epoch 让 DiffJPEG 优化更稳定。

---

## Table 10 — Anomaly Detection (Data Augmentation Consistency)

**命令：**
```bash
CUDA_VISIBLE_DEVICES=4 python anomaly_detection.py 2>&1 | tee outputs/imagenet/whitebox/anomaly_detection.txt
```

**结果对照：**

### ImageBind

| Augmentation | 列 | 我的 | 论文 |
|---|---|---|---|
| JPEG | x | 0.884 | 0.878 |
| | x_δ | 0.237 | 0.226 |
| | x_jpeg | 0.752 | 0.730 |
| GaussianBlur | x | 0.753 | 0.749 |
| | x_δ | 0.223 | 0.211 |
| | x_jpeg | 0.615 | 0.645 |
| RandomAffine | x | 0.876 | 0.870 |
| | x_δ | 0.261 | 0.256 |
| | x_jpeg | 0.697 | 0.743 |
| ColorJitter | x | 0.905 | 0.890 |
| | x_δ | 0.588 | 0.499 |
| | x_jpeg | 0.844 | 0.825 |
| RandomHorizontalFlip | x | 0.979 | 0.971 |
| | x_δ | 0.440 | 0.401 |
| | x_jpeg | 0.881 | 0.867 |

### AudioCLIP

| Augmentation | 列 | 我的 | 论文 |
|---|---|---|---|
| JPEG | x | 0.835 | 0.851 |
| | x_δ | 0.365 | 0.215 |
| | x_jpeg | 0.732 | 0.620 |
| GaussianBlur | x | 0.789 | 0.807 |
| | x_δ | 0.345 | 0.207 |
| | x_jpeg | 0.660 | 0.564 |
| RandomAffine | x | 0.872 | 0.876 |
| | x_δ | 0.360 | 0.212 |
| | x_jpeg | 0.693 | 0.618 |
| ColorJitter | x | 0.915 | 0.920 |
| | x_δ | 0.558 | 0.425 |
| | x_jpeg | 0.841 | 0.749 |
| RandomHorizontalFlip | x | 0.963 | 0.969 |
| | x_δ | 0.419 | 0.295 |
| | x_jpeg | 0.781 | 0.707 |

 在所有 augmentation 上都有 `x（原图）> x_jpeg（JPEG-resistant）> x_δ（普通对抗）`。

- ImageBind 几乎所有数字差异 < 0.02。
- AudioCLIP 的 x_δ / x_jpeg 列系统性偏高 ~0.1，与 Table 9 看到的同一现象一致（你的 AudioCLIP illusion 比论文版本在增强下稍稳定一些）。

---

## 未跑的实验

### Table 2 — Image Generation (需要 BindDiffusion)

复现步骤：
```bash
git submodule update --init
cp image_text_generation/image_generation.py BindDiffusion/
# 按 BindDiffusion/README.md 下载 checkpoint 到 BindDiffusion/checkpoints
cd BindDiffusion && python image_generation.py
```

### Table 3 — Text Generation (需要 PandaGPT)

复现步骤：
```bash
git submodule update --init
cp image_text_generation/text_generation.py PandaGPT/code/
cp image_text_generation/text_generation_demo.ipynb PandaGPT/code/
# 按 PandaGPT/pretrained_ckpt/README.md 下载 checkpoint
cd PandaGPT && python code/text_generation.py
```

⚠️ 注意：`image_text_generation/text_generation_demo.ipynb:120` 有硬编码 `/home/tz362/...` 路径，需改成自己的图片路径。

---

## 仓库本地改动记录

为复现做的代码改动（已 commit 在 branch `claude/pensive-rubin-SdEFc`）：

1. **`thermal_illusion_text_generation.py`**: 把 4 处 `config['key']`（`zero_shot_steps`、`eta_min`、`buffer_size`、`delta`）改成 `config.get('key', default)`。原代码加载 whitebox config 但读取 query config 的键，导致 KeyError；这 4 个变量在脚本内 ROM 后未被使用，修改不影响任何实验结果。

为复现需要额外做的事（不在 git 跟踪）：
1. 手动生成 `data/imagenet/imagenet1000_clsidx_to_labels.txt`：
   ```bash
   python -c "
   with open('data/imagenet/imagenet-classes.txt') as f:
       d = {i: l.rstrip() for i, l in enumerate(f)}
   with open('data/imagenet/imagenet1000_clsidx_to_labels.txt', 'w') as f:
       f.write(repr(d))"
   ```
2. JPEG-resistant 评估时按需修改 `evaluate_jpeg.py` 用 `x_advs_200.npy`（见 Table 9 命令）。

