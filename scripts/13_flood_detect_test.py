"""
=============================================================================
13_flood_detect_test.py — 内涝检测模型推理测试
=============================================================================

使用训练好的 best.pt 模型对真实场景图片做内涝检测推理，
输出每张图片的检测结果并生成可视化标注图。

【运行方式】
    conda activate yolov8-py310
    python scripts/13_flood_detect_test.py
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "flood_detect" / "weights" / "best.pt"
TEST_DIR = PROJECT_ROOT / "datasets" / "flood" / "test2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flood_test"
CONFIDENCE_THRESHOLD = 0.25  # 置信度阈值
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  城市内涝检测 — 真实场景推理测试")
    print("=" * 60)

    # 加载模型
    print(f"\n模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 获取测试图片
    test_images = sorted(
        f for f in TEST_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"测试图片: {len(test_images)} 张")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    # 逐张推理
    total_detections = 0
    results_summary = []

    for img_path in test_images:
        print(f"{'─' * 50}")
        print(f"图片: {img_path.name}")

        # 读取原图获取尺寸
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
        boxes = result.boxes

        num_detections = len(boxes)
        total_detections += num_detections

        # 分析每个检测框
        flood_total_area = 0
        det_details = []

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / img_area * 100
            flood_total_area += box_area

            det_details.append({
                "cls": cls_name,
                "conf": conf,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "area_ratio": area_ratio,
            })

            print(f"  检测 #{i + 1}: {cls_name} | 置信度: {conf:.2f} | "
                  f"面积占比: {area_ratio:.1f}% | "
                  f"位置: ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")

        # 积水覆盖率
        flood_coverage = flood_total_area / img_area * 100
        if flood_coverage > 100:
            flood_coverage = min(flood_coverage, 100)  # 重叠框可能超100%

        # 内涝等级判定
        if num_detections == 0:
            level = "无内涝"
            max_conf = 0.0
        else:
            max_conf = max(d["conf"] for d in det_details)
            if flood_coverage < 10:
                level = "轻度积水"
            elif flood_coverage < 35:
                level = "中度内涝"
            else:
                level = "严重内涝"

        print(f"  → 检测框数: {num_detections} | 积水覆盖: {flood_coverage:.1f}% | "
              f"最高置信度: {max_conf:.2f} | 等级: 【{level}】")

        results_summary.append({
            "image": img_path.name,
            "detections": num_detections,
            "max_conf": max_conf,
            "coverage": flood_coverage,
            "level": level,
            "details": det_details,
        })

        # 保存标注后的图片
        annotated = result.plot()
        output_path = OUTPUT_DIR / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)

    # ===================================================================
    # 汇总报告
    # ===================================================================
    print(f"\n{'=' * 60}")
    print(f"  推理测试汇总报告")
    print(f"{'=' * 60}")
    print(f"\n{'图片':<16} {'检测数':>6} {'最高置信度':>10} {'积水覆盖':>10} {'内涝等级':<10}")
    print(f"{'─' * 60}")
    for r in results_summary:
        print(f"{r['image']:<16} {r['detections']:>6} {r['max_conf']:>10.2f} "
              f"{r['coverage']:>9.1f}% {r['level']:<10}")

    detected_count = sum(1 for r in results_summary if r["detections"] > 0)
    print(f"{'─' * 60}")
    print(f"总计: {len(results_summary)} 张图片, "
          f"{detected_count} 张检出内涝, "
          f"{total_detections} 个检测框")

    avg_conf = 0
    if total_detections > 0:
        all_confs = [d["conf"] for r in results_summary for d in r["details"]]
        avg_conf = sum(all_confs) / len(all_confs)
    print(f"平均置信度: {avg_conf:.2f}")

    # 等级分布
    level_counts = {}
    for r in results_summary:
        level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1
    print(f"\n等级分布:")
    for level, count in level_counts.items():
        print(f"  {level}: {count} 张")

    print(f"\n标注结果已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
