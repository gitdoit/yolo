"""
=============================================================================
04_custom_train.py — 自定义训练入门
=============================================================================

前面的脚本都是用"预训练模型"（别人训练好的）来检测通用物体。
但如果你要检测的东西不在预训练模型的 80 个类别中（比如：特定产品缺陷、自定义标志等），
就需要自己训练一个模型。

本脚本演示：
    1. 使用 COCO128 迷你数据集（COCO 的前 128 张图）
    2. 对预训练模型做"微调"（Transfer Learning / Fine-tuning）
    3. 训练完成后用新模型做检测

【Java 工程师理解方式】
    预训练模型 = 一个已经学会了"通用视觉能力"的基座
    微调训练   = 在此基础上，用你自己的数据"教"它识别特定的东西
    类比：就像一个有经验的 Java 工程师（预训练），到新公司后只需学习业务知识（微调），
          而不用从零学编程。

【运行方式】
    conda activate yolov8-py310
    python scripts/04_custom_train.py

【注意】
    - 训练需要 GPU，RTX 4070 SUPER 完全够用
    - COCO128 数据集首次运行会自动下载（约 7MB）
    - 训练 5 个 epoch 大约需要几分钟
    - 训练结果保存在 outputs/custom_train/
"""

from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 训练参数
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"  # 基于 nano 模型微调
DATASET = "coco128.yaml"       # Ultralytics 内置的迷你数据集配置
EPOCHS = 5                     # 训练轮数（入门演示用 5 轮，实际项目通常 50~300 轮）
IMAGE_SIZE = 640               # 输入图片尺寸（YOLO 默认 640x640）
BATCH_SIZE = 8                 # 每批处理的图片数量（显存不够可以调小）


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # 第1步: 加载预训练模型
    # ===================================================================
    print("=" * 60)
    print("  自定义训练入门 — 使用 COCO128 迷你数据集")
    print("=" * 60)

    print(f"\n[1/3] 加载预训练模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # ===================================================================
    # 第2步: 开始训练
    # ===================================================================
    # model.train() 会：
    #   1. 自动下载 COCO128 数据集
    #   2. 在数据集上训练指定轮数
    #   3. 保存最佳模型和最终模型
    #   4. 生成训练指标图表
    #
    # 类比 Java:
    #   model.train() 就像 Maven 的 mvn package —— 一个命令完成"编译+测试+打包"
    #   这里是 "数据加载+模型训练+评估+保存" 一条龙

    print(f"[2/3] 开始训练 (epochs={EPOCHS}, batch={BATCH_SIZE}, imgsize={IMAGE_SIZE})")
    print("    首次运行会下载 COCO128 数据集（约 7MB）...")
    print("    训练过程中的指标：")
    print("      · box_loss: 检测框位置的误差（越小越好）")
    print("      · cls_loss: 物体分类的误差（越小越好）")
    print("      · mAP50:    平均精度@IoU=0.5（越大越好，1.0 = 完美）\n")

    results = model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(OUTPUT_DIR),
        name="custom_train",
        exist_ok=True,
        device=0,           # 使用 GPU 0（你的 RTX 4070 SUPER）
        verbose=True,
        amp=False,           # 跳过 AMP 检查（该检查需要联网下载模型，容易超时）
    )

    # ===================================================================
    # 第3步: 使用训练后的模型检测
    # ===================================================================
    print(f"\n[3/3] 训练完成！使用新模型做检测验证...")

    # 训练后的最佳模型路径
    best_model_path = OUTPUT_DIR / "custom_train" / "weights" / "best.pt"
    print(f"最佳模型: {best_model_path}")

    if best_model_path.exists():
        # 加载训练后的模型
        trained_model = YOLO(str(best_model_path))

        # 用训练后的模型做检测（用 COCO128 中的图片做验证）
        val_results = trained_model.val(
            data=DATASET,
            project=str(OUTPUT_DIR),
            name="custom_train_val",
            exist_ok=True,
            workers=0,          # Windows 下避免多进程内存不足
        )
        print(f"\n验证结果:")
        print(f"  mAP50:    {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")

    print("\n" + "=" * 60)
    print("  训练流程回顾")
    print("=" * 60)
    print("""
    完整的训练流程：

    ┌───────────────┐
    │  准备数据集    │  ← 收集图片 + 标注（画框标记物体位置和类别）
    └───────┬───────┘
            ↓
    ┌───────────────┐
    │  选择基础模型  │  ← 通常用预训练模型做起点（迁移学习）
    └───────┬───────┘
            ↓
    ┌───────────────┐
    │  配置并训练    │  ← model.train(data=..., epochs=...)
    └───────┬───────┘
            ↓
    ┌───────────────┐
    │  评估模型      │  ← model.val() 查看精度指标
    └───────┬───────┘
            ↓
    ┌───────────────┐
    │  部署使用      │  ← model.predict() 或 导出为 ONNX/TensorRT
    └───────────────┘

    在实际项目中，你最需要投入的是【准备数据集】这一步。
    推荐工具：
    - LabelImg: 简单易用的图片标注工具
    - Roboflow: 在线标注 + 数据集管理平台
    - CVAT:     功能丰富的开源标注平台
    """)


if __name__ == "__main__":
    main()
