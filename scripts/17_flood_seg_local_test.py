"""
=============================================================================
17_flood_seg_local_test.py — 本地单张图片积水分割测试
=============================================================================

使用训练好的 best.pt 对本地测试图片进行积水分割推理，
输出积水区域数、覆盖率、置信度及内涝等级判定。

【运行方式】
    conda activate yolov8-py310
    python scripts/17_flood_seg_local_test.py
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flood_seg_local_test"
CONFIDENCE_THRESHOLD = 0.25

# 测试图片列表（可追加更多路径）
TEST_IMAGES = [
    Path(r"D:\workspace\company\vision-repo\data\files\test\2026-03-31\9d01e65d-e9d3-4885-985e-2050635a95ee.jpg"),
]

# 内涝等级阈值（基于积水面积占比）
LEVEL_THRESHOLDS = {
    "light": 5,       # <5% 微量, 5~15% 轻度
    "moderate": 15,    # 15~35% 中度
    "severe": 35,      # >35% 严重
}


def classify_level(num_detections: int, water_coverage: float) -> str:
    """根据检测数量和覆盖率判定内涝等级"""
    if num_detections == 0:
        return "无积水"
    if water_coverage < LEVEL_THRESHOLDS["light"]:
        return "微量积水"
    if water_coverage < LEVEL_THRESHOLDS["moderate"]:
        return "轻度积水"
    if water_coverage < LEVEL_THRESHOLDS["severe"]:
        return "中度内涝"
    return "严重内涝"


def process_image(model: YOLO, img_path: Path, output_dir: Path):
    """对单张图片进行推理并输出结果"""
    print(f"\n{'─' * 55}")
    print(f"图片: {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print("  ⚠ 无法读取图片，跳过")
        return None

    h, w = img.shape[:2]
    img_area = h * w
    print(f"  尺寸: {w} x {h}")

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

    # 计算积水面积
    water_pixel_count = 0
    num_detections = 0
    max_conf = 0.0

    if masks is not None and len(masks) > 0:
        num_detections = len(masks)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for i, mask in enumerate(masks.data):
            conf = float(boxes.conf[i])
            max_conf = max(max_conf, conf)
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(
                mask_np, (w, h), interpolation=cv2.INTER_NEAREST
            )
            combined_mask = np.maximum(
                combined_mask, (mask_resized > 0.5).astype(np.uint8)
            )
        water_pixel_count = int(combined_mask.sum())

    water_coverage = water_pixel_count / img_area * 100
    level = classify_level(num_detections, water_coverage)

    print(f"  区域数: {num_detections}")
    print(f"  最高置信度: {max_conf:.2f}")
    print(f"  积水覆盖率: {water_coverage:.1f}%")
    print(f"  内涝等级: 【{level}】")

    # 保存标注图片
    annotated = result.plot()
    output_path = output_dir / f"result_{img_path.stem}.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"  标注图片已保存: {output_path}")

    return {
        "image": img_path.name,
        "detections": num_detections,
        "max_conf": max_conf,
        "coverage": water_coverage,
        "level": level,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  积水分割模型 — 本地测试")
    print("=" * 60)

    # 加载模型
    if not MODEL_PATH.exists():
        print(f"\n  ⚠ 模型文件不存在: {MODEL_PATH}")
        return

    print(f"\n模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 验证图片
    valid_images = []
    for p in TEST_IMAGES:
        if p.exists():
            valid_images.append(p)
        else:
            print(f"  ⚠ 图片不存在，已跳过: {p}")

    print(f"有效测试图片: {len(valid_images)} 张")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"输出目录: {OUTPUT_DIR}")

    if not valid_images:
        print("\n  没有可测试的图片")
        return

    # 逐张推理
    results_summary = []
    for img_path in valid_images:
        r = process_image(model, img_path, OUTPUT_DIR)
        if r:
            results_summary.append(r)

    # 汇总
    if results_summary:
        print(f"\n{'=' * 60}")
        print(f"  测试汇总")
        print(f"{'=' * 60}")
        print(f"\n{'图片':<40} {'区域':>4} {'置信度':>6} {'覆盖率':>7} {'等级':<8}")
        print(f"{'─' * 70}")
        for r in results_summary:
            short = r["image"][:38] if len(r["image"]) > 38 else r["image"]
            print(
                f"{short:<40} {r['detections']:>4} "
                f"{r['max_conf']:>6.2f} {r['coverage']:>6.1f}% {r['level']:<8}"
            )

        detected = [r for r in results_summary if r["detections"] > 0]
        print(f"{'─' * 70}")
        print(f"总计: {len(results_summary)} 张, {len(detected)} 张检出积水")
        if detected:
            avg_cov = np.mean([r["coverage"] for r in detected])
            avg_conf = np.mean([r["max_conf"] for r in detected])
            print(f"检出图片平均覆盖率: {avg_cov:.1f}%")
            print(f"检出图片平均置信度: {avg_conf:.2f}")

    print(f"\n结果已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
