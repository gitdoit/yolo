"""
=============================================================================
15_flood_seg_test.py — 城市积水分割模型推理测试
=============================================================================

使用训练好的 YOLOv8n-seg 模型对图片做积水区域分割推理。
基于分割掩码精确计算积水覆盖面积，输出内涝等级判定。

适用场景：接入城市摄像头，雨天抓取图片后自动识别道路积水。

与 13_flood_detect_test.py 的区别：
    13 是目标检测（只有检测框，面积估算粗糙）
    15 是实例分割（像素级掩码，面积计算更精确）

【运行方式】
    conda activate yolov8-py310
    python scripts/15_flood_seg_test.py

【测试图片】
    默认使用 datasets/flood1/test/images 下的图片。
    也可替换 TEST_DIR 为你自己的摄像头抓图目录。
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "flood_seg" / "weights" / "best.pt"
TEST_DIR = PROJECT_ROOT / "datasets" / "flood1" / "test" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flood_seg_test"
CONFIDENCE_THRESHOLD = 0.25  # 置信度阈值
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 内涝等级阈值（基于积水面积占比）
LEVEL_THRESHOLDS = {
    "light": 5,     # <5% 无积水, 5~15% 轻度
    "moderate": 15,  # 15~35% 中度
    "severe": 35,    # >35% 严重
}


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  城市积水分割 — 推理测试")
    print("=" * 60)

    # 加载模型
    if not MODEL_PATH.exists():
        print(f"\n  ❌ 模型文件不存在: {MODEL_PATH}")
        print(f"  请先运行 14_flood_seg_train.py 训练模型")
        return

    print(f"\n模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 获取测试图片
    if not TEST_DIR.exists():
        print(f"\n  ❌ 测试目录不存在: {TEST_DIR}")
        return

    test_images = sorted(
        f for f in TEST_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not test_images:
        print(f"\n  ⚠️ 测试目录中没有图片: {TEST_DIR}")
        return

    # 只取前 20 张做演示（数据集测试集有几百张）
    max_images = 20
    if len(test_images) > max_images:
        print(f"测试集共 {len(test_images)} 张图片，取前 {max_images} 张做演示")
        test_images = test_images[:max_images]
    else:
        print(f"测试图片: {len(test_images)} 张")

    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    # 逐张推理
    results_summary = []

    for img_path in test_images:
        print(f"{'─' * 55}")
        print(f"图片: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠ 无法读取图片，跳过")
            continue

        h, w = img.shape[:2]
        img_area = h * w

        # 推理
        results = model.predict(
            source=str(img_path),
            conf=CONFIDENCE_THRESHOLD,
            iou=0.5,
            verbose=False,
        )

        result = results[0]
        masks = result.masks
        boxes = result.boxes

        # 计算积水面积（基于分割掩码，比检测框更精确）
        water_pixel_count = 0
        num_detections = 0
        max_conf = 0.0

        if masks is not None and len(masks) > 0:
            num_detections = len(masks)

            # 合并所有水域掩码（避免重叠区域重复计算）
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for i, mask in enumerate(masks.data):
                conf = float(boxes.conf[i])
                max_conf = max(max_conf, conf)
                # masks.data 是 (N, mask_h, mask_w) tensor，需要 resize 到原图尺寸
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, (w, h), interpolation=cv2.INTER_NEAREST
                )
                combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8))

            water_pixel_count = int(combined_mask.sum())

        water_coverage = water_pixel_count / img_area * 100

        # 内涝等级判定
        if num_detections == 0:
            level = "无积水"
        elif water_coverage < LEVEL_THRESHOLDS["light"]:
            level = "微量积水"
        elif water_coverage < LEVEL_THRESHOLDS["moderate"]:
            level = "轻度积水"
        elif water_coverage < LEVEL_THRESHOLDS["severe"]:
            level = "中度内涝"
        else:
            level = "严重内涝"

        print(f"  检测区域数: {num_detections} | 积水覆盖: {water_coverage:.1f}% | "
              f"最高置信度: {max_conf:.2f} | 等级: 【{level}】")

        results_summary.append({
            "image": img_path.name,
            "detections": num_detections,
            "max_conf": max_conf,
            "coverage": water_coverage,
            "level": level,
        })

        # 保存标注后的图片（含分割掩码可视化）
        annotated = result.plot()
        output_path = OUTPUT_DIR / f"seg_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)

    # ===================================================================
    # 汇总报告
    # ===================================================================
    print(f"\n{'=' * 60}")
    print(f"  积水分割推理汇总报告")
    print(f"{'=' * 60}")
    print(f"\n{'图片':<30} {'区域数':>6} {'置信度':>8} {'覆盖率':>8} {'等级':<10}")
    print(f"{'─' * 65}")
    for r in results_summary:
        name = r["image"][:28] if len(r["image"]) > 28 else r["image"]
        print(f"{name:<30} {r['detections']:>6} {r['max_conf']:>8.2f} "
              f"{r['coverage']:>7.1f}% {r['level']:<10}")

    detected_count = sum(1 for r in results_summary if r["detections"] > 0)
    print(f"{'─' * 65}")
    print(f"总计: {len(results_summary)} 张图片, {detected_count} 张检出积水")

    if detected_count > 0:
        avg_coverage = sum(r["coverage"] for r in results_summary if r["detections"] > 0) / detected_count
        avg_conf = sum(r["max_conf"] for r in results_summary if r["detections"] > 0) / detected_count
        print(f"检出图片平均覆盖率: {avg_coverage:.1f}%")
        print(f"检出图片平均置信度: {avg_conf:.2f}")

    # 等级分布
    level_counts = {}
    for r in results_summary:
        level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1
    print(f"\n等级分布:")
    for level_name, count in level_counts.items():
        print(f"  {level_name}: {count} 张")

    print(f"\n分割结果已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
