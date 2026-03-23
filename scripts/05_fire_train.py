"""
=============================================================================
05_fire_train.py — 火焰检测自定义训练
=============================================================================

使用从 Roboflow 下载的 Fire 数据集，训练一个火焰/烟雾检测模型。

数据集信息：
    - 来源: Roboflow (Fire dataset)
    - 类别: exist(存在火源), fire(火焰), smoke(烟雾)
    - 数据量: 训练集 121 张, 验证集 53 张, 测试集 21 张

与 04_custom_train.py 的区别：
    04 是用 Ultralytics 内置的 COCO128（自动下载，80 个类别，演示性质）
    05 是用你从 Roboflow 下载的自定义数据集（3 个类别，真正的"自己的数据"）

【关键概念 — data.yaml】
    data.yaml 是 YOLO 训练的"数据集说明书"，里面定义了：
      · train/val/test 图片路径
      · nc: 类别数量
      · names: 每个类别的名称

    类比 Java：data.yaml 就像 application.yml，
    告诉框架去哪找数据、数据长什么样。

【运行方式】
    conda activate yolov8-py310
    python scripts/05_fire_train.py

【注意】
    - 训练需要 GPU，RTX 4070 SUPER 完全够用
    - 训练 30 个 epoch 大约需要几分钟
    - 训练结果保存在 outputs/fire_train/
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
DATASET_DIR = PROJECT_ROOT / "datasets" / "Fire"
DATA_YAML = DATASET_DIR / "data.yaml"

# 训练参数
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"  # 基于 nano 模型微调
EPOCHS = 30                  # 训练轮数（数据集小，30 轮足够看到效果）
IMAGE_SIZE = 640             # 输入图片尺寸
BATCH_SIZE = 8               # 每批处理图片数（显存不够可调小为 4）


# ============================================================================
# 辅助函数
# ============================================================================

def fix_data_yaml():
    """
    修正 data.yaml 中的路径问题。

    Roboflow 导出的 data.yaml 使用相对路径（如 ../train/images），
    这在不同工作目录下运行时可能出错。
    我们将其改为绝对路径，确保无论从哪里运行都能正确找到数据。

    类比 Java：就像把 pom.xml 中的相对路径改为绝对路径，避免 CI 环境报错。
    """
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 将相对路径改为绝对路径
    data["train"] = str(DATASET_DIR / "train" / "images")
    data["val"] = str(DATASET_DIR / "valid" / "images")
    data["test"] = str(DATASET_DIR / "test" / "images")

    # 写入修正后的 yaml（保存到输出目录，不修改原始文件）
    fixed_yaml = OUTPUT_DIR / "fire_train" / "data.yaml"
    fixed_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(fixed_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"  原始 data.yaml 路径: {DATA_YAML}")
    print(f"  修正后保存到:       {fixed_yaml}")
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
    print("  火焰检测模型训练 — 使用 Roboflow Fire 数据集")
    print("=" * 60)

    # ===================================================================
    # 第1步: 检查数据集
    # ===================================================================
    print(f"\n[1/5] 检查数据集...")

    if not DATASET_DIR.exists():
        print(f"  ❌ 数据集目录不存在: {DATASET_DIR}")
        print(f"  请将 Fire 数据集放到 datasets/Fire/ 目录下")
        return

    # 统计各子集图片数量
    train_count = len(list((DATASET_DIR / "train" / "images").glob("*")))
    valid_count = len(list((DATASET_DIR / "valid" / "images").glob("*")))
    test_count = len(list((DATASET_DIR / "test" / "images").glob("*")))
    print(f"  训练集: {train_count} 张图片")
    print(f"  验证集: {valid_count} 张图片")
    print(f"  测试集: {test_count} 张图片")

    # ===================================================================
    # 第2步: 修正 data.yaml 路径
    # ===================================================================
    print(f"\n[2/5] 修正 data.yaml 路径...")
    fixed_yaml = fix_data_yaml()

    # ===================================================================
    # 第3步: 加载预训练模型
    # ===================================================================
    print(f"\n[3/5] 加载预训练模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("  ✓ 模型加载成功")
    print("  模型架构: YOLOv8n (nano)")
    print("  原始训练: COCO 数据集 (80 类)")
    print("  本次任务: 微调为火焰检测 (3 类: exist, fire, smoke)")

    # ===================================================================
    # 第4步: 开始训练
    # ===================================================================
    print(f"\n[4/5] 开始训练")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  批大小:   {BATCH_SIZE}")
    print(f"  图片尺寸: {IMAGE_SIZE}")
    print()
    print("  训练过程中关注这些指标：")
    print("    · box_loss: 检测框定位误差 → 越小越好")
    print("    · cls_loss: 分类误差       → 越小越好")
    print("    · dfl_loss: 分布焦点损失   → 越小越好")
    print("    · mAP50:    精度@IoU=0.5   → 越大越好 (1.0=完美)")
    print()

    results = model.train(
        data=str(fixed_yaml),       # 使用修正后的 data.yaml
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(OUTPUT_DIR),
        name="fire_train",
        exist_ok=True,
        device=0,                   # 使用 GPU 0（你的 RTX 4070 SUPER）
        verbose=True,
        amp=False,                  # 跳过 AMP 检查
        workers=0,                  # Windows 下避免多进程问题
        patience=10,                # 早停：如果 10 轮没有提升就停止
    )

    # ===================================================================
    # 第5步: 评估与测试
    # ===================================================================
    print(f"\n[5/5] 训练完成！开始评估...")

    best_model_path = OUTPUT_DIR / "fire_train" / "weights" / "best.pt"
    print(f"  最佳模型: {best_model_path}")

    if best_model_path.exists():
        # 加载训练好的模型
        trained_model = YOLO(str(best_model_path))

        # --- 在验证集上评估 ---
        print("\n  [评估] 在验证集上计算指标...")
        val_results = trained_model.val(
            data=str(fixed_yaml),
            project=str(OUTPUT_DIR),
            name="fire_train_val",
            exist_ok=True,
            workers=0,
        )
        print(f"\n  验证结果:")
        print(f"    mAP50:    {val_results.box.map50:.4f}")
        print(f"    mAP50-95: {val_results.box.map:.4f}")

        # --- 在测试集图片上做预测，直观看效果 ---
        print("\n  [预测] 用测试集图片做预测，保存可视化结果...")
        test_images_dir = DATASET_DIR / "test" / "images"
        test_images = list(test_images_dir.glob("*"))[:5]  # 取前 5 张做预测

        if test_images:
            predict_results = trained_model.predict(
                source=[str(img) for img in test_images],
                project=str(OUTPUT_DIR),
                name="fire_train_predict",
                exist_ok=True,
                save=True,              # 保存带标注的图片
                conf=0.25,              # 置信度阈值
            )

            print(f"\n  预测结果已保存到: {OUTPUT_DIR / 'fire_train_predict'}")
            for r in predict_results:
                boxes = r.boxes
                print(f"    {Path(r.path).name}: 检测到 {len(boxes)} 个目标", end="")
                if len(boxes) > 0:
                    # 显示每个检测到的类别和置信度
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = r.names[cls_id]
                        print(f"  [{cls_name} {conf:.2f}]", end="")
                print()

    # ===================================================================
    # 总结
    # ===================================================================
    print("\n" + "=" * 60)
    print("  训练完成！文件说明")
    print("=" * 60)
    print(f"""
    输出目录: {OUTPUT_DIR / 'fire_train'}

    重要文件：
    ├── weights/
    │   ├── best.pt          ← 验证集上表现最好的模型（部署用这个）
    │   └── last.pt          ← 最后一轮的模型
    ├── results.csv          ← 每轮训练的详细指标数据
    ├── results.png          ← 训练曲线图（loss、mAP 随 epoch 的变化）
    ├── confusion_matrix.png ← 混淆矩阵（看模型把什么类搞混了）
    ├── F1_curve.png         ← F1 分数曲线
    ├── PR_curve.png         ← 精确率-召回率曲线
    └── val_batch*.jpg       ← 验证集预测可视化

    下一步建议：
    1. 查看 results.png，观察 loss 是否持续下降、mAP 是否持续上升
    2. 查看 confusion_matrix.png，了解模型容易混淆哪些类别
    3. 查看 fire_train_predict/ 目录下的预测图片，直观感受效果
    4. 如果效果不好，可以：
       · 增加 EPOCHS（比如 50~100）
       · 增加数据量（在 Roboflow 上扩充数据集）
       · 尝试更大的模型（yolov8s.pt 或 yolov8m.pt）
    """)


if __name__ == "__main__":
    main()
