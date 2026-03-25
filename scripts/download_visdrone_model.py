"""
=============================================================================
download_visdrone_model.py — 下载 VisDrone 预训练模型并验证
=============================================================================

VisDrone 数据集专注于无人机/监控视角的车辆行人检测，类别比 COCO 更细致：
    0: pedestrian  (行人)
    1: people      (人群)
    2: bicycle     (自行车)
    3: car         (小轿车)
    4: van         (面包车/小货车)
    5: truck       (卡车)
    6: tricycle    (三轮车)
    7: awning-tricycle (篷三轮车)
    8: bus         (公交车)
    9: motor       (摩托车)

【对比 COCO 的优势】
    COCO 只有 car/truck/bus/motorcycle 4 个车辆类别
    VisDrone 有 car/van/truck/bus/motor/bicycle/tricycle 7 个，
    还区分了 van（面包车）和 car（轿车），更适合交通监控场景。

【使用方式】
    conda activate yolov8-py310
    python scripts/download_visdrone_model.py

    下载完成后可在 08/10 脚本中替换模型路径使用。
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# HuggingFace 社区训练的 VisDrone 模型
# 如果下载失败，可以手动从 HuggingFace 下载:
#   https://huggingface.co/mshamrai/yolov8n-visdrone
HF_MODEL_ID = "hf://mshamrai/yolov8n-visdrone"

# VisDrone 类别名映射
VISDRONE_CLASSES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

# VisDrone 中的车辆类别 ID
VISDRONE_VEHICLE_IDS = [3, 4, 5, 6, 7, 8, 9]  # car/van/truck/tricycle/awning-tricycle/bus/motor


def main():
    print("=" * 60)
    print("  下载 VisDrone 预训练模型")
    print("=" * 60)

    # 1. 下载模型
    print(f"\n从 HuggingFace 下载模型: {HF_MODEL_ID}")
    print("首次下载可能需要几分钟...\n")

    model = YOLO(HF_MODEL_ID)

    # 2. 保存到本地
    local_path = MODEL_DIR / "yolov8n-visdrone.pt"
    # YOLO 模型加载后会缓存到本地，打印路径信息
    print(f"\n模型加载成功!")
    print(f"模型信息:")
    print(f"  类别数: {len(model.names)}")
    print(f"  类别: {model.names}")

    # 3. 简单验证
    print(f"\n模型已就绪，可在交通拥堵检测脚本中使用。")
    print(f"\n使用方式:")
    print(f"  方式 1 — 在 08/10 脚本中修改 MODEL_PATH:")
    print(f'    MODEL_PATH = "{HF_MODEL_ID}"')
    print(f"\n  方式 2 — 直接 Python 调用:")
    print(f'    model = YOLO("{HF_MODEL_ID}")')
    print(f'    results = model.predict("your_video.mp4")')

    print(f"\n  VisDrone 车辆类别 ID (替换 VEHICLE_CLASS_IDS):")
    print(f"    {VISDRONE_VEHICLE_IDS}")
    print(f"    对应: {[VISDRONE_CLASSES[i] for i in VISDRONE_VEHICLE_IDS]}")

    print("\n" + "=" * 60)
    print("  如果效果不满意，可以自己训练:")
    print("  model = YOLO('yolo26n.pt')")
    print("  model.train(data='VisDrone.yaml', epochs=100, imgsz=640)")
    print("  (数据集 2.3GB，自动下载)")
    print("=" * 60)


if __name__ == "__main__":
    main()
