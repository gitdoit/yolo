"""
=============================================================================
14_flood_seg_train.py — 城市积水语义分割模型训练
=============================================================================

基于 Roboflow 的 teste_water_seg 数据集（flood1），使用 YOLO26n-seg 训练
积水区域分割模型。后续可部署到城市摄像头，在雨天自动识别道路积水区域。

数据集信息：
    - 来源: Roboflow (teste_water_seg v1)
    - 类别: 1 类 (water)
    - 格式: YOLO Segmentation（多边形标注）
    - 数据量: 训练集 ~5900 张 / 验证集 ~1700 张 / 测试集 ~840 张
    - 预处理: Auto-Orient

与 12_flood_detect_train.py 的区别：
    12 是目标检测（画检测框） — 用的 flood 数据集
    14 是实例分割（像素级轮廓） — 用的 flood1 数据集，效果更精细

【运行方式】
    conda activate yolov8-py310
    python scripts/14_flood_seg_train.py

【训练完成后】
    - 最佳模型: outputs/flood_seg/weights/best.pt
    - 训练曲线: outputs/flood_seg/ 下各种 png 图
    - 可用 best.pt 做积水区域分割推理
"""

import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 数据集路径
DATASET_DIR = PROJECT_ROOT / "datasets" / "flood1"
DATA_YAML = DATASET_DIR / "data.yaml"

# 训练参数
MODEL_NAME = "yolo26n-seg.pt"    # YOLO26 nano 分割模型（自动下载）
EPOCHS = 80                       # 训练轮数（数据量大，80 轮可充分收敛）
IMAGE_SIZE = 640                  # 输入图片尺寸
BATCH_SIZE = 8                    # RTX 4070 SUPER 12GB，分割模型显存占用较大


# ============================================================================
# 辅助函数
# ============================================================================

def fix_data_yaml():
    """
    修正 data.yaml 中的路径为绝对路径。

    Roboflow 导出的 data.yaml 使用相对路径（如 ../train/images），
    改为绝对路径确保在任何工作目录下都能正确运行。
    修正后的文件保存到输出目录，不修改原始文件。
    """
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 将相对路径改为绝对路径
    data["train"] = str(DATASET_DIR / "train" / "images")
    data["val"] = str(DATASET_DIR / "valid" / "images")
    data["test"] = str(DATASET_DIR / "test" / "images")

    # 写入修正后的 yaml（保存到输出目录，不修改原始文件）
    fixed_yaml = OUTPUT_DIR / "flood_seg" / "data.yaml"
    fixed_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(fixed_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"  原始 data.yaml: {DATA_YAML}")
    print(f"  修正后保存到:   {fixed_yaml}")
    print(f"  训练集路径: {data['train']}")
    print(f"  验证集路径: {data['val']}")
    print(f"  测试集路径: {data['test']}")
    print(f"  类别数量: {data['nc']}")
    print(f"  类别名称: {data['names']}")

    return fixed_yaml


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  城市积水分割模型训练 — YOLO26n-seg + flood1 数据集")
    print("=" * 60)

    # ===================================================================
    # 第1步: 检查数据集
    # ===================================================================
    print(f"\n[1/5] 检查数据集...")

    if not DATASET_DIR.exists():
        print(f"  ❌ 数据集目录不存在: {DATASET_DIR}")
        print(f"  请将 flood1 数据集放到 datasets/flood1/ 目录下")
        return

    # 统计各子集图片数量
    train_count = len(list((DATASET_DIR / "train" / "images").glob("*")))
    valid_count = len(list((DATASET_DIR / "valid" / "images").glob("*")))
    test_count = len(list((DATASET_DIR / "test" / "images").glob("*")))
    print(f"  训练集: {train_count} 张图片")
    print(f"  验证集: {valid_count} 张图片")
    print(f"  测试集: {test_count} 张图片")
    print(f"  总计:   {train_count + valid_count + test_count} 张图片")

    # ===================================================================
    # 第2步: 修正 data.yaml 路径
    # ===================================================================
    print(f"\n[2/5] 修正 data.yaml 路径...")
    fixed_yaml = fix_data_yaml()

    # ===================================================================
    # 第3步: 加载预训练模型
    # ===================================================================
    print(f"\n[3/5] 加载预训练分割模型: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    print("  ✓ 模型加载成功")
    print("  模型架构: YOLO26n-seg (nano segmentation)")
    print("  原始训练: COCO 数据集 (80 类实例分割)")
    print("  本次任务: 微调为积水区域分割 (1 类: water)")

    # ===================================================================
    # 第4步: 开始训练
    # ===================================================================
    print(f"\n[4/5] 开始训练")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  批大小:   {BATCH_SIZE}")
    print(f"  图片尺寸: {IMAGE_SIZE}")
    print()
    print("  训练过程中关注这些指标：")
    print("    · box_loss:  检测框定位误差     → 越小越好")
    print("    · seg_loss:  分割掩码误差       → 越小越好（分割任务核心指标）")
    print("    · cls_loss:  分类误差           → 越小越好")
    print("    · mAP50:     精度@IoU=0.5       → 越大越好")
    print("    · mAP50-95:  精度@IoU=0.5~0.95  → 越大越好（综合评价）")
    print()

    results = model.train(
        data=str(fixed_yaml),       # 使用修正后的 data.yaml
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(OUTPUT_DIR),
        name="flood_seg",
        exist_ok=True,
        device=0,                   # 使用 GPU 0（RTX 4070 SUPER）
        verbose=True,
        amp=False,                  # 跳过 AMP 检查避免联网超时
        workers=0,                  # Windows 下避免多进程问题
        patience=15,                # 早停：15 轮没有提升就停止
        save=True,
        save_period=10,             # 每 10 轮保存一次 checkpoint
        plots=True,                 # 生成训练曲线图

        # 数据增强（针对积水场景优化）
        hsv_h=0.015,                # 色调扰动（水面反光颜色变化）
        hsv_s=0.5,                  # 饱和度扰动（雨天光线暗淡）
        hsv_v=0.3,                  # 亮度扰动（阴雨天气）
        degrees=5.0,                # 小角度旋转（摄像头安装角度差异）
        translate=0.1,              # 平移
        scale=0.3,                  # 缩放（远近不同的积水区域）
        fliplr=0.5,                 # 水平翻转
        flipud=0.0,                 # 不做垂直翻转（不符合实际场景）
        mosaic=1.0,                 # Mosaic 增强
        mixup=0.1,                  # MixUp 增强
    )

    # ===================================================================
    # 第5步: 验证模型
    # ===================================================================
    print(f"\n[5/5] 训练完成！开始验证...")

    best_model_path = OUTPUT_DIR / "flood_seg" / "weights" / "best.pt"
    print(f"  最佳模型路径: {best_model_path}")

    if best_model_path.exists():
        trained_model = YOLO(str(best_model_path))

        val_results = trained_model.val(
            data=str(fixed_yaml),
            project=str(OUTPUT_DIR),
            name="flood_seg_val",
            exist_ok=True,
        )
        print(f"\n  验证结果:")
        print(f"    Box  mAP50:    {val_results.box.map50:.4f}")
        print(f"    Box  mAP50-95: {val_results.box.map:.4f}")
        print(f"    Mask mAP50:    {val_results.seg.map50:.4f}")
        print(f"    Mask mAP50-95: {val_results.seg.map:.4f}")
    else:
        print("  ⚠️ 未找到最佳模型文件，跳过验证")

    print(f"\n{'=' * 60}")
    print(f"  训练完成！")
    print(f"  最佳模型: outputs/flood_seg/weights/best.pt")
    print(f"  后续使用: python scripts/15_flood_seg_test.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
