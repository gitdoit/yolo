"""
=============================================================================
20_clip_flood_benchmark.py — CLIP 零样本图片分类性能测试
=============================================================================

用途：
  - 用 OpenCLIP 做零样本 (zero-shot) 图片分类
  - 对比"洪水/积水"与"正常场景"两组文本描述的相似度
  - 评估 CLIP 方案作为 VLM 替代的可行性

原理：
  CLIP 将图片和文字映射到同一向量空间，计算余弦相似度。
  不需要训练，不需要标注数据，直接用自然语言描述即可分类。

保存内容：
  - results.csv        每张图片的分类结果
  - summary.json       汇总统计

【运行方式】
  conda activate yolov8
  python scripts/20_clip_flood_benchmark.py
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

# HuggingFace 国内镜像 — 必须在 import open_clip 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from PIL import Image

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "datasets" / "flood" / "split_train" / "images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "clip_benchmark"

# 模型选择 — 按性能/速度权衡
# ViT-B-32: 快速，精度一般 (~400MB)
# ViT-L-14: 精度更高，稍慢 (~900MB)
# ViT-H-14: 最高精度，较大 (~2.5GB)
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

MAX_IMAGES = 20
RESIZE_MAX = 0  # CLIP 自带预处理，设为 0 不额外缩放

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================================
# 文本提示模板 — 核心：正面 vs 负面描述
# ============================================================================

# 正面 = 有洪水/积水的描述
FLOOD_POSITIVE_TEXTS = [
    "a photo of a flooded street",
    "a photo of standing water on the road",
    "a photo of a flood with water covering the ground",
    "a photo of a river overflowing its banks",
    "a photo of flood water surrounding buildings",
    "a photo of a waterlogged area after heavy rain",
]

# 负面 = 正常场景描述
FLOOD_NEGATIVE_TEXTS = [
    "a photo of a normal dry street",
    "a photo of a clear road with no water",
    "a photo of a sunny day with dry ground",
    "a photo of a clean city street",
    "a photo of a normal river within its banks",
    "a photo of a dry construction site",
]


def collect_images(image_dir: Path, max_images: int) -> list[Path]:
    """收集图片列表。"""
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:max_images]


def run_benchmark(
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output_root: Path = OUTPUT_ROOT,
    force_cpu: bool = False,
) -> None:
    import open_clip

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("CLIP 零样本洪水分类测试")
    print("=" * 72)
    print(f"模型: {CLIP_MODEL} ({CLIP_PRETRAINED})")
    print(f"设备: {device}")
    print(f"正面描述数: {len(FLOOD_POSITIVE_TEXTS)}")
    print(f"负面描述数: {len(FLOOD_NEGATIVE_TEXTS)}")
    print(f"图片目录: {image_dir}")

    # ---- 加载模型 ----
    print("正在加载 CLIP 模型...")
    t0 = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"模型加载完成 ({load_time:.2f}s)")

    # ---- 预编码文本 ----
    all_texts = FLOOD_POSITIVE_TEXTS + FLOOD_NEGATIVE_TEXTS
    text_tokens = tokenizer(all_texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    n_pos = len(FLOOD_POSITIVE_TEXTS)
    n_neg = len(FLOOD_NEGATIVE_TEXTS)

    # ---- 收集图片 ----
    images = collect_images(image_dir, MAX_IMAGES)
    print(f"图片数量: {len(images)}")

    # ---- 输出目录 ----
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {run_dir}")

    # ---- 推理 ----
    results = []
    wall_times = []

    for idx, img_path in enumerate(images, 1):
        print("-" * 72)
        print(f"[{idx:02d}/{len(images):02d}] {img_path.name}")

        t_start = time.perf_counter()

        # 加载并预处理
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # 编码图片
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarities = (image_features @ text_features.T).squeeze(0).cpu().tolist()

        pos_scores = similarities[:n_pos]
        neg_scores = similarities[n_pos:]
        avg_pos = sum(pos_scores) / n_pos
        avg_neg = sum(neg_scores) / n_neg

        # 判定：正面平均分 > 负面平均分 → 判为洪水
        is_flood = avg_pos > avg_neg
        confidence = avg_pos - avg_neg  # 差值越大越确定

        wall_time = time.perf_counter() - t_start
        wall_times.append(wall_time)

        result = {
            "image": img_path.name,
            "is_flood": is_flood,
            "avg_positive": round(avg_pos, 4),
            "avg_negative": round(avg_neg, 4),
            "confidence_gap": round(confidence, 4),
            "wall_seconds": round(wall_time, 4),
            "top_positive": round(max(pos_scores), 4),
            "top_negative": round(max(neg_scores), 4),
        }
        results.append(result)

        label = "FLOOD" if is_flood else "NORMAL"
        print(
            f"  {label} | gap={confidence:+.4f} | "
            f"pos={avg_pos:.4f} neg={avg_neg:.4f} | "
            f"{wall_time:.4f}s"
        )

    # ---- 写 CSV ----
    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # ---- 汇总 ----
    flood_true = sum(1 for r in results if r["is_flood"])
    flood_false = len(results) - flood_true

    summary = {
        "model": f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
        "device": device,
        "image_count": len(results),
        "positive_texts": FLOOD_POSITIVE_TEXTS,
        "negative_texts": FLOOD_NEGATIVE_TEXTS,
        "wall_seconds": {
            "avg": round(statistics.mean(wall_times), 4),
            "min": round(min(wall_times), 4),
            "max": round(max(wall_times), 4),
            "p50": round(statistics.median(wall_times), 4),
            "p95": round(sorted(wall_times)[int(len(wall_times) * 0.95)], 4)
            if len(wall_times) >= 2
            else round(max(wall_times), 4),
        },
        "model_load_seconds": round(load_time, 2),
        "flood_true_count": flood_true,
        "flood_false_count": flood_false,
        "avg_confidence_gap": round(
            statistics.mean(r["confidence_gap"] for r in results), 4
        ),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- 打印汇总 ----
    print("=" * 72)
    print("测试完成")
    print(f"判定为洪水: {flood_true} / {len(results)}")
    print(f"判定为正常: {flood_false} / {len(results)}")
    print(f"平均耗时: {summary['wall_seconds']['avg']:.4f}s")
    print(f"模型加载: {load_time:.2f}s")
    print(f"设备: {device}")
    print(f"结果目录: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()
    run_benchmark(force_cpu=args.cpu)
