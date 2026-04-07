# CLIP 零样本洪水分类 — 工作原理说明

> 对应脚本：[scripts/20_clip_flood_benchmark.py](../scripts/20_clip_flood_benchmark.py)

---

## 核心思路

CLIP（Contrastive Language–Image Pretraining）将 **图片** 和 **文字** 映射到同一个高维向量空间。  
分类时不需要训练，只需要提供正面/负面的自然语言描述，计算图片向量与每条文字向量的 **余弦相似度**，取平均后比较大小即可判定类别。

$$
\text{similarity}(I, T) = \frac{\vec{v}_I \cdot \vec{v}_T}{\|\vec{v}_I\| \cdot \|\vec{v}_T\|}
$$

---

## 数据流

```
输入图片目录
    │
    ▼
[PIL 加载 → CLIP 预处理] ──► 图片向量 v_I  (归一化)
                                    │
                                    │  余弦相似度 @ text_features.T
                                    ▼
文字描述 (正面 6 条 + 负面 6 条)
    │
    ▼
[Tokenizer → CLIP 文本编码器] ──► 文字向量矩阵  (归一化，预计算一次)
                                    │
                                    ▼
                           similarities[12]
                           ┌──────────────────┐
                           │ pos_scores[0..5] │  → avg_pos
                           │ neg_scores[6..11]│  → avg_neg
                           └──────────────────┘
                                    │
                           avg_pos > avg_neg ?
                           ├─ 是 → FLOOD ✓
                           └─ 否 → NORMAL ✗
```

---

## 输入

### 图片目录

```
datasets/flood/split_train/images/
```

示例文件（按字母序取前 20 张）：

| # | 文件名 |
|---|--------|
| 1 | [dirty_120_jpg.rf.f58b8645f541f02498335d1b6f38c538.jpg](../datasets/flood/split_train/images/dirty_120_jpg.rf.f58b8645f541f02498335d1b6f38c538.jpg) |
| 2 | [flood_100_jpg.rf.d8aaa0adf274c8f0b179eb9531077577.jpg](../datasets/flood/split_train/images/flood_100_jpg.rf.d8aaa0adf274c8f0b179eb9531077577.jpg) |
| 3 | [flood_102_jpg.rf.27ae3b5320609ef78d1b0655d48b6fb3.jpg](../datasets/flood/split_train/images/flood_102_jpg.rf.27ae3b5320609ef78d1b0655d48b6fb3.jpg) |
| 4 | [flood_104_jpg.rf.914e0cf4c6d869324c9b098b52ac7855.jpg](../datasets/flood/split_train/images/flood_104_jpg.rf.914e0cf4c6d869324c9b098b52ac7855.jpg) |
| … | （共 20 张，MAX_IMAGES=20） |

> 支持格式：`.jpg` `.jpeg` `.png` `.bmp` `.webp`

### 文本提示（Prompt Templates）

**正面描述（Flood Positive，6 条）**

```
a photo of a flooded street
a photo of standing water on the road
a photo of a flood with water covering the ground
a photo of a river overflowing its banks
a photo of flood water surrounding buildings
a photo of a waterlogged area after heavy rain
```

**负面描述（Normal Negative，6 条）**

```
a photo of a normal dry street
a photo of a clear road with no water
a photo of a sunny day with dry ground
a photo of a clean city street
a photo of a normal river within its banks
a photo of a dry construction site
```

### 模型配置

| 参数 | 值 |
|------|----|
| 模型 | `ViT-B-32` |
| 预训练权重 | `openai` |
| 设备 | `cpu`（无 GPU 时自动回退） |
| 最大图片数 | `20` |

---

## 每张图片的推理步骤

1. `Image.open(img_path).convert("RGB")` — 加载图片并转 RGB
2. `preprocess(pil_img).unsqueeze(0)` — CLIP 内置预处理（resize 224×224、归一化）
3. `model.encode_image(img_tensor)` — 提取图片特征向量，L2 归一化
4. `image_features @ text_features.T` — 与 12 条文字向量做点积（即余弦相似度）
5. 分割前 6 / 后 6，各自取平均 → `avg_pos`、`avg_neg`
6. `avg_pos > avg_neg` → 判为 **FLOOD**，否则 **NORMAL**
7. `confidence_gap = avg_pos - avg_neg`（越大越确定）

---

## 输出

### 目录结构

```
outputs/clip_benchmark/
└── run_<YYYYMMDD_HHMMSS>/
    ├── results.csv       ← 每张图片的分类结果
    └── summary.json      ← 整体统计摘要
```

### results.csv — 逐图结果

| 字段 | 说明 |
|------|------|
| `image` | 图片文件名 |
| `is_flood` | 是否判定为洪水（True/False） |
| `avg_positive` | 6 条正面描述相似度均值 |
| `avg_negative` | 6 条负面描述相似度均值 |
| `confidence_gap` | `avg_pos - avg_neg`，正值越大越确信是洪水 |
| `wall_seconds` | 单张推理耗时（秒） |
| `top_positive` | 正面描述中最高相似度 |
| `top_negative` | 负面描述中最高相似度 |

**实际运行示例（run_20260407_153505，前 5 行）**

| image | is_flood | avg_positive | avg_negative | confidence_gap | wall_seconds |
|-------|----------|-------------|-------------|---------------|--------------|
| dirty_120_…jpg | False | 0.2144 | 0.2398 | -0.0254 | 0.0834 |
| flood_100_…jpg | True  | 0.2770 | 0.2246 |  0.0524 | 0.0869 |
| flood_102_…jpg | True  | 0.2720 | 0.2266 |  0.0454 | 0.0911 |
| flood_104_…jpg | True  | 0.2668 | 0.2254 |  0.0414 | 0.0922 |
| flood_105_…jpg | True  | 0.2712 | 0.2263 |  0.0450 | 0.0864 |

### summary.json — 整体统计

```json
{
  "model": "ViT-B-32/openai",
  "device": "cpu",
  "image_count": 20,
  "wall_seconds": {
    "avg": 0.0806,
    "min": 0.0551,
    "max": 0.0922,
    "p50": 0.0849,
    "p95": 0.0922
  },
  "model_load_seconds": 2.95,
  "flood_true_count": 19,
  "flood_false_count": 1,
  "avg_confidence_gap": 0.0424
}
```

---

## 性能观测（CPU，20 张图）

| 指标 | 值 |
|------|----|
| 模型加载 | 2.95 s（一次性） |
| 单张平均推理 | **0.08 s** |
| 20 张总推理 | ~1.6 s |
| 判定为洪水 | 19 / 20 |
| 判定为正常 | 1 / 20（dirty_120，泥泞场景误判） |
| 平均置信度差 | 0.0424 |

> 误判分析：`dirty_120` 为泥泞/脏乱场景，与"正常干燥街道"比"洪水"更接近，gap = -0.0254，说明现有正面 prompt 对泥泞场景区分能力较弱。

---

## 关键源码位置速查

| 功能 | 位置 |
|------|------|
| 模型加载 | `scripts/20_clip_flood_benchmark.py` L99–106 |
| 文字预编码 | L109–112 |
| 图片推理循环 | L127–163 |
| 判定逻辑 | L148–151 |
| CSV 写入 | L166–171 |
| summary 生成 | L174–199 |
