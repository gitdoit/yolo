"""
=============================================================================
12_flood_detect_train.py — 城市内涝检测模型训练
=============================================================================

基于 Roboflow 的 Flood 目标检测数据集，使用 YOLO26n 训练内涝检测模型。
数据集只有 train 集，脚本会自动拆分为 train/val（85%/15%）。

数据集信息：
    - 来源: Roboflow (flood-vhyqu)
    - 类别: 1 类 (Flood)
    - 格式: YOLOv8 (图片 + txt 标签)

【运行方式】
    conda activate yolov8-py310
    python scripts/12_flood_detect_train.py

【训练完成后】
    - 最佳模型: outputs/flood_detect/weights/best.pt
    - 训练指标图表: outputs/flood_detect/ 下的各种 png 图
    - 可直接用 best.pt 做推理检测
"""

import random
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
DATASET_DIR = PROJECT_ROOT / "datasets" / "flood"
ORIGINAL_TRAIN_IMAGES = DATASET_DIR / "train" / "images"
ORIGINAL_TRAIN_LABELS = DATASET_DIR / "train" / "labels"

# 模型
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n.pt"

# 训练参数
EPOCHS = 100            # 训练轮数（数据量不大，100轮足够收敛）
IMAGE_SIZE = 640        # 输入图片尺寸
BATCH_SIZE = 16         # RTX 4070 SUPER 12GB 跑 YOLO26n 可以用 16
VAL_SPLIT = 0.15        # 验证集比例
RANDOM_SEED = 42        # 随机种子，保证可复现


# ============================================================================
# 数据集拆分
# ============================================================================

def split_dataset():
    """
    将原始 train 集拆分为 train/val 两个子集。
    如果已经拆分过（split_images/split_labels 目录存在），则跳过。
    """
    split_train_images = DATASET_DIR / "split_train" / "images"
    split_train_labels = DATASET_DIR / "split_train" / "labels"
    split_val_images = DATASET_DIR / "split_val" / "images"
    split_val_labels = DATASET_DIR / "split_val" / "labels"

    # 检查是否已拆分
    if split_train_images.exists() and split_val_images.exists():
        train_count = len(list(split_train_images.glob("*")))
        val_count = len(list(split_val_images.glob("*")))
        if train_count > 0 and val_count > 0:
            print(f"  数据集已拆分: train={train_count}, val={val_count}，跳过拆分")
            return split_train_images.parent.parent

    print("  开始拆分数据集...")

    # 获取所有图片文件（不含扩展名的文件名）
    image_files = sorted(ORIGINAL_TRAIN_IMAGES.glob("*"))
    image_stems = [f.stem for f in image_files]
    image_ext_map = {f.stem: f.suffix for f in image_files}

    # 随机打乱
    random.seed(RANDOM_SEED)
    random.shuffle(image_stems)

    # 拆分
    val_count = int(len(image_stems) * VAL_SPLIT)
    val_stems = set(image_stems[:val_count])
    train_stems = set(image_stems[val_count:])

    print(f"  总图片数: {len(image_stems)}")
    print(f"  训练集: {len(train_stems)}, 验证集: {len(val_stems)}")

    # 创建目录
    for d in [split_train_images, split_train_labels, split_val_images, split_val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # 复制文件
    for stem in train_stems:
        ext = image_ext_map[stem]
        shutil.copy2(ORIGINAL_TRAIN_IMAGES / f"{stem}{ext}", split_train_images / f"{stem}{ext}")
        label_file = ORIGINAL_TRAIN_LABELS / f"{stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, split_train_labels / f"{stem}.txt")

    for stem in val_stems:
        ext = image_ext_map[stem]
        shutil.copy2(ORIGINAL_TRAIN_IMAGES / f"{stem}{ext}", split_val_images / f"{stem}{ext}")
        label_file = ORIGINAL_TRAIN_LABELS / f"{stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, split_val_labels / f"{stem}.txt")

    print("  拆分完成！")
    return DATASET_DIR


def create_data_yaml():
    """
    生成指向拆分后数据的 data.yaml 配置文件。
    使用绝对路径，避免相对路径问题。
    """
    yaml_path = DATASET_DIR / "flood_train.yaml"

    data = {
        "path": str(DATASET_DIR),
        "train": "split_train/images",
        "val": "split_val/images",
        "nc": 1,
        "names": ["Flood"],
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"  数据配置文件: {yaml_path}")
    return yaml_path


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  城市内涝检测 — YOLO26n 模型训练")
    print("=" * 60)

    # ===================================================================
    # 第1步: 拆分数据集
    # ===================================================================
    print(f"\n[1/4] 拆分数据集 (train:{1 - VAL_SPLIT:.0%} / val:{VAL_SPLIT:.0%})")
    split_dataset()

    # ===================================================================
    # 第2步: 生成数据配置
    # ===================================================================
    print(f"\n[2/4] 生成训练配置文件")
    data_yaml = create_data_yaml()

    # ===================================================================
    # 第3步: 训练模型
    # ===================================================================
    print(f"\n[3/4] 开始训练")
    print(f"  模型: {MODEL_PATH.name}")
    print(f"  参数: epochs={EPOCHS}, batch={BATCH_SIZE}, imgsz={IMAGE_SIZE}")
    print(f"  训练指标说明:")
    print(f"    · box_loss: 检测框定位误差（越小越好）")
    print(f"    · cls_loss: 分类误差（越小越好）")
    print(f"    · mAP50:    IoU=0.5 时的平均精度（越大越好）")
    print(f"    · mAP50-95: IoU=0.5~0.95 的平均精度（越大越好）\n")

    model = YOLO(str(MODEL_PATH))

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(OUTPUT_DIR),
        name="flood_detect",
        exist_ok=True,
        device=0,
        verbose=True,
        amp=False,              # 跳过 AMP 检查避免联网超时
        workers=4,              # 数据加载线程数
        patience=20,            # 早停：20 轮无提升则停止
        save=True,              # 保存每轮权重
        save_period=10,         # 每 10 轮保存一次 checkpoint
        plots=True,             # 生成训练曲线图

        # 数据增强（针对内涝场景优化）
        hsv_h=0.015,            # 色调扰动（水面颜色变化）
        hsv_s=0.5,              # 饱和度扰动（雨天光线变化大）
        hsv_v=0.3,              # 亮度扰动（阴暗天气）
        degrees=5.0,            # 小角度旋转（摄像头安装角度差异）
        translate=0.1,          # 平移
        scale=0.3,              # 缩放（远近不同的积水）
        fliplr=0.5,             # 水平翻转
        flipud=0.0,             # 不做垂直翻转（不符合实际场景）
        mosaic=1.0,             # Mosaic 增强（拼接4张图，增加小目标多样性）
        mixup=0.1,              # MixUp 增强（轻度混合）
    )

    # ===================================================================
    # 第4步: 验证模型
    # ===================================================================
    print(f"\n[4/4] 模型验证")

    best_model_path = OUTPUT_DIR / "flood_detect" / "weights" / "best.pt"
    print(f"  最佳模型: {best_model_path}")

    if best_model_path.exists():
        trained_model = YOLO(str(best_model_path))

        val_results = trained_model.val(
            data=str(data_yaml),
            project=str(OUTPUT_DIR),
            name="flood_detect_val",
            exist_ok=True,
            workers=0,
        )

        print(f"\n  验证结果:")
        print(f"    mAP50:    {val_results.box.map50:.4f}")
        print(f"    mAP50-95: {val_results.box.map:.4f}")
        print(f"    Precision: {val_results.box.mp:.4f}")
        print(f"    Recall:    {val_results.box.mr:.4f}")
    else:
        print("  未找到训练后的模型文件，请检查训练日志。")

    # ===================================================================
    # 完成
    # ===================================================================
    print("\n" + "=" * 60)
    print("  训练完成！")
    print("=" * 60)
    print(f"""
    模型文件:
      best.pt  → {OUTPUT_DIR / "flood_detect" / "weights" / "best.pt"}
      last.pt  → {OUTPUT_DIR / "flood_detect" / "weights" / "last.pt"}

    使用方式:
      from ultralytics import YOLO
      model = YOLO("outputs/flood_detect/weights/best.pt")
      results = model.predict("your_image.jpg")

    下一步:
      · 用 best.pt 对实际摄像头抓拍图片做推理测试
      · 如果效果不好，可增加数据或调整训练参数
      · 可导出为 ONNX/TensorRT 格式部署到生产环境
    """)


if __name__ == "__main__":
    main()
