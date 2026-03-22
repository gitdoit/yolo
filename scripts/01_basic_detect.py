"""
=============================================================================
01_basic_detect.py — Hello World：图片物体检测
=============================================================================

这是你的第一个 YOLO 程序！
它会加载一个预训练模型，对图片进行物体检测，并显示/保存结果。

【Java 工程师理解方式】
    想象你在写一个图片分析的 REST 接口：
    1. 加载模型  → 类比 Spring Boot 启动时初始化 Service Bean
    2. 传入图片  → 类比 接收 HTTP 请求中的图片参数
    3. 返回结果  → 类比 返回 JSON 响应，包含检测框坐标和类别

【运行方式】
    conda activate yolov8-py310
    python scripts/01_basic_detect.py

【首次运行说明】
    - 会自动下载 yolov8n.pt 模型文件（约 6MB）
    - 会自动下载一张示例图片用于测试
    - 检测结果保存在 outputs/ 目录
"""

import os
import urllib.request
from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 配置区 — 类比 Java 的 application.yml
# ============================================================================

# 项目根目录（自动计算，无需手动修改）
PROJECT_ROOT = Path(__file__).parent.parent

# 模型选择 — YOLO 提供 5 种预训练模型大小：
#   yolov8n.pt  → Nano   (最小最快, 3.2M参数, 适合入门/边缘设备)  ← 我们用这个
#   yolov8s.pt  → Small  (11.2M参数)
#   yolov8m.pt  → Medium (25.9M参数)
#   yolov8l.pt  → Large  (43.7M参数)
#   yolov8x.pt  → XLarge (最大最准, 68.2M参数, 适合服务器部署)
#
# 类比 Java: 就像选择不同规格的服务器配置 —— n是开发机, x是生产机
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"

# 输入输出路径
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 检测参数
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值（0~1），只显示置信度>50%的检测结果
# 类比 Java: 类似于搜索引擎的相关性阈值，低于此值的结果不返回


# ============================================================================
# 辅助函数
# ============================================================================

def download_sample_image():
    """下载 Ultralytics 官方示例图片用于测试"""
    sample_path = IMAGES_DIR / "bus.jpg"
    if not sample_path.exists():
        print("正在下载示例图片 bus.jpg ...")
        url = "https://ultralytics.com/images/bus.jpg"
        urllib.request.urlretrieve(url, str(sample_path))
        print(f"示例图片已保存到: {sample_path}")
    return sample_path


def print_detection_results(results):
    """
    解析并打印检测结果 — 帮助你理解返回的数据结构

    【数据结构类比 Java】
    results          → List<DetectionFrame>    一帧/一张图的结果列表
    results[0]       → DetectionFrame          第一张图的结果
    results[0].boxes → List<BoundingBox>       该图中所有检测框
    每个 box 包含:
        .xyxy   → float[4]   检测框坐标 [左上x, 左上y, 右下x, 右下y]
        .conf   → float      置信度 (0~1)，模型认为这是某物体的概率
        .cls    → int        类别编号（0=person, 1=bicycle, 2=car, ...共80类）
    """
    result = results[0]  # 我们只传了一张图，所以取第一个结果

    print("\n" + "=" * 60)
    print(f"  检测完成！共发现 {len(result.boxes)} 个物体")
    print("=" * 60)

    # 获取类别名称映射（类比 Java 的 Map<Integer, String>）
    class_names = result.names  # {0: 'person', 1: 'bicycle', 2: 'car', ...}

    for i, box in enumerate(result.boxes):
        # 提取每个检测框的信息
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 检测框坐标
        confidence = box.conf[0].item()          # 置信度
        class_id = int(box.cls[0].item())        # 类别编号
        class_name = class_names[class_id]       # 类别名称

        print(f"  [{i+1}] {class_name:12s}  "
              f"置信度: {confidence:.1%}  "
              f"位置: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")

    print("=" * 60)


# ============================================================================
# 主流程 — 核心只有 3 步！
# ============================================================================

def main():
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # --- 准备测试图片 ---
    # 优先使用 data/images/ 下已有的图片，否则下载示例图
    existing_images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    if existing_images:
        image_path = existing_images[0]
        print(f"使用已有图片: {image_path}")
    else:
        image_path = download_sample_image()

    # =======================================================================
    # ★ 核心代码 — 就这 3 行！
    # =======================================================================

    # 第1步: 加载模型
    # 类比 Java: YoloService service = new YoloService("yolov8n.pt");
    # 模型文件存放在 models/ 目录下
    print(f"\n[1/3] 加载模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 第2步: 执行推理（检测）
    # 类比 Java: List<Detection> results = service.detect(imageBytes);
    # conf 参数: 只返回置信度 > 阈值的结果
    print(f"[2/3] 执行检测: {image_path.name}")
    results = model.predict(
        source=str(image_path),     # 输入图片路径
        conf=CONFIDENCE_THRESHOLD,  # 置信度阈值
        save=True,                  # 自动保存标注后的图片
        project=str(OUTPUT_DIR),    # 输出目录
        name="basic_detect",        # 输出子目录名
        exist_ok=True,              # 目录已存在时不报错
    )

    # 第3步: 查看结果
    print("[3/3] 检测结果:")
    print_detection_results(results)

    # 保存路径提示
    output_path = OUTPUT_DIR / "basic_detect" / image_path.name
    print(f"\n标注图片已保存到: {output_path}")
    print("提示: 也可以调用 results[0].show() 弹窗查看（需要图形界面）")


# ============================================================================
# Python 入口点 — 类比 Java 的 public static void main(String[] args)
# ============================================================================
if __name__ == "__main__":
    main()
