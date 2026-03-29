"""
=============================================================================
16_flood_seg_validate.py — 使用 water_test 数据集验证积水分割模型
=============================================================================

从 datasets/water_test/test/images 中挑选多种场景的图片，
用训练好的 YOLO26n-seg 模型进行推理，评估模型泛化能力。

挑选覆盖以下场景：
    - 城市CCTV监控画面（雅加达街道积水监控）
    - 飓风洪水场景（Houston Harvey）
    - 河流监控（芒加莱、芝利翁河、门腾）
    - 欧洲洪水（易北河汉堡泛滥）
    - 其他积水/洪灾场景

【运行方式】
    conda activate yolov8-py310
    python scripts/16_flood_seg_validate.py
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
TEST_DIR = PROJECT_ROOT / "datasets" / "water_test" / "test" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flood_seg_validate"
CONFIDENCE_THRESHOLD = 0.25
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 内涝等级阈值（基于积水面积占比）
LEVEL_THRESHOLDS = {
    "light": 5,      # <5% 微量, 5~15% 轻度
    "moderate": 15,   # 15~35% 中度
    "severe": 35,     # >35% 严重
}

# 精选 20 张测试图片 — 覆盖不同场景
SELECTED_IMAGES = [
    # ---- 城市CCTV监控 (雅加达街道积水) ----
    "501521_JKS_SDA_JL-PONDOK-KARYA-C01_CCTV-01_09-08_02-03-2024_jpg.rf.7878281cc77f33499f1e121261b8f549.jpg",
    "501522_JKS_SDA_JL-PUTRI-MUTIARA-V-C01_CCTV-02_15-32_02-03-2024_jpg.rf.addf52143c12dc90d5e5478f1f681559.jpg",
    "501529_JKS_SDA_JL-H-IPIN-C01_CCTV-02_08-21_03-03-2024_jpg.rf.4a99a7e62b894f6b9f2b8c115433f1d9.jpg",
    "501531_JKT_SDA_JL-BALAI-RAKYAT-C01_CCTV-02_15-35_03-03-2024_jpg.rf.24ec1b8ff79e6206e81b4049df740fa1.jpg",
    "501509_JKP_SDA_JL-CEMPAKA-PUTIH-BARAT-C01_CCTV-02_13-29_03-03-2024_jpg.rf.fddbfb322362ed198170828d71e17e4b.jpg",
    "501539_JKU_SDA_TAMAN-PEGANGSAAN-INDAH-C01_CCTV-01_12-29_03-03-2024_jpg.rf.3c02f10c221c13f1b070b285e21ed85e.jpg",
    "501516_JKS_SDA_GG-KOBA-C01_CCTV-01_09-08_02-03-2024_jpg.rf.4387f607e9d7c500c00d572453024f3f.jpg",
    "501601_JKT_KESBANGPOL_JL-RAYA-KALIBATA-C01_CCTV-01_08-21_03-03-2024_jpg.rf.65151d3de3f3ccdcdb5c043dfc297242.jpg",
    # ---- 飓风/洪灾场景 ----
    "10020_Hurricane-Harvey-Flood-Houston-TX-Meyerland-Neighborhood-August-27-2017-Garage-Time-Lapse-ZOpWO7rJbtU-_jpg.rf.5ba1eccc5dbd5b2f88f1a60099959631.jpg",
    "210_Hurricane-Harvey-flooding-time-lapse-_bg9Fy0uCOI-_jpg.rf.d16722c12bf09c90908952b9f87af047.jpg",
    "300_2023-Flood-Time-Lapse-VNzrOBGzkA4-_jpg.rf.b876cfe2adaf4e5769dc7d1ab43ff8b1.jpg",
    # ---- 河流水位监控 ----
    "manggarai_2023Y_11M_17D_14H_jpg.rf.260b5bcec590a1f133021bd04c51d712.jpg",
    "ciliwung_2023Y_11M_17D_14H_jpg.rf.d4a877cfb23319891b0ddd52b4eca408.jpg",
    "srengseng-sawah_2023Y_11M_18D_14H_jpg.rf.f386ee7c522541018b6419049426af80.jpg",
    "menteng_2023Y_11M_18D_13H_jpg.rf.074a621ebecb291dad8fc35da01c6718.jpg",
    # ---- 欧洲洪水（易北河泛滥） ----
    "Time-Lapse-Shows-River-Elbe-Overflowing-and-Flooding-its-Banks-in-Hamburg-1104587_mp4-54_jpg.rf.5f315f2db7a13f095d542d24934fdb59.jpg",
    # ---- 其他监控/积水场景 ----
    "20160917_133703_jpg.rf.7a774cacbd7844e6a584779721e51b36.jpg",
    "20201113_085415-SHOP2_jpg.rf.9a19f48d83ee406d3f26b094c8b46e17.jpg",
    "frame237_jpg.rf.4abccbc2df1be10be8c9f3378cf3fc27.jpg",
    "20230911_182238-scaled_jpg.rf.2dfae7742cd3f897f190ca9c07f4387b.jpg",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  积水分割模型 — water_test 数据集验证")
    print("=" * 60)

    # 加载模型
    if not MODEL_PATH.exists():
        print(f"\n  模型文件不存在: {MODEL_PATH}")
        return

    print(f"\n模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 验证选中的图片是否存在
    test_images = []
    missing = []
    for name in SELECTED_IMAGES:
        p = TEST_DIR / name
        if p.exists():
            test_images.append(p)
        else:
            missing.append(name)

    if missing:
        print(f"\n  以下 {len(missing)} 张图片未找到，已跳过:")
        for m in missing:
            print(f"    - {m[:60]}...")

    print(f"\n有效测试图片: {len(test_images)} 张")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    if not test_images:
        print("  没有可测试的图片")
        return

    # 逐张推理
    results_summary = []

    for img_path in test_images:
        print(f"{'─' * 55}")
        # 简短的场景标识
        name = img_path.name
        if "CCTV" in name:
            scene = "[CCTV监控]"
        elif "Hurricane" in name or "Harvey" in name:
            scene = "[飓风洪灾]"
        elif "Flood" in name or "flood" in name:
            scene = "[洪水场景]"
        elif "manggarai" in name or "ciliwung" in name or "menteng" in name or "srengseng" in name:
            scene = "[河流监控]"
        elif "Elbe" in name or "Hamburg" in name:
            scene = "[易北河洪水]"
        else:
            scene = "[其他场景]"

        short_name = name[:50] + "..." if len(name) > 50 else name
        print(f"{scene} {short_name}")

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

        print(f"  区域数: {num_detections} | 覆盖: {water_coverage:.1f}% | "
              f"置信度: {max_conf:.2f} | 等级: 【{level}】")

        results_summary.append({
            "image": img_path.name,
            "scene": scene,
            "detections": num_detections,
            "max_conf": max_conf,
            "coverage": water_coverage,
            "level": level,
        })

        # 保存标注图片
        annotated = result.plot()
        output_path = OUTPUT_DIR / f"val_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), annotated)

    # ===================================================================
    # 汇总报告
    # ===================================================================
    print(f"\n{'=' * 60}")
    print(f"  water_test 验证汇总报告")
    print(f"{'=' * 60}")
    print(f"\n{'场景':<12} {'图片':<32} {'区域':>4} {'置信度':>6} {'覆盖率':>7} {'等级':<8}")
    print(f"{'─' * 75}")
    for r in results_summary:
        short = r["image"][:30] if len(r["image"]) > 30 else r["image"]
        print(f"{r['scene']:<12} {short:<32} {r['detections']:>4} "
              f"{r['max_conf']:>6.2f} {r['coverage']:>6.1f}% {r['level']:<8}")

    detected_count = sum(1 for r in results_summary if r["detections"] > 0)
    print(f"{'─' * 75}")
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

    # 按场景统计
    scene_stats = {}
    for r in results_summary:
        s = r["scene"]
        if s not in scene_stats:
            scene_stats[s] = {"total": 0, "detected": 0, "coverages": []}
        scene_stats[s]["total"] += 1
        if r["detections"] > 0:
            scene_stats[s]["detected"] += 1
            scene_stats[s]["coverages"].append(r["coverage"])

    print(f"\n按场景统计:")
    for s, stats in scene_stats.items():
        det_rate = stats["detected"] / stats["total"] * 100
        avg_cov = np.mean(stats["coverages"]) if stats["coverages"] else 0
        print(f"  {s}: {stats['total']}张, 检出率{det_rate:.0f}%, 平均覆盖{avg_cov:.1f}%")

    print(f"\n验证结果已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
