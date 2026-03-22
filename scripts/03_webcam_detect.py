"""
=============================================================================
03_webcam_detect.py — 摄像头实时物体检测
=============================================================================

用你电脑的摄像头进行实时物体检测！
这是最能体现 YOLO 名称含义的 demo：
    YOLO = You Only Look Once（你只需看一次）
    → 一次前向传播（一次"看"）就能同时检测出图中所有物体
    → 所以速度极快，可以做到实时检测

【Java 工程师理解方式】
    这就像一个"实时流处理系统"：
    while (camera.isOpen()) {
        Frame frame = camera.capture();
        List<Detection> detections = model.predict(frame);
        display(drawBoxes(frame, detections));
    }

【运行方式】
    conda activate yolov8-py310
    python scripts/03_webcam_detect.py

【操作说明】
    - 运行后会打开摄像头预览窗口
    - 窗口中会实时标注检测到的物体
    - 按 'q' 键退出
    - 如果没有摄像头，脚本会提示错误
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================================
# 配置区
# ============================================================================

MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n.pt"  # nano 模型，推理最快
CAMERA_INDEX = 0               # 摄像头编号（0=默认摄像头, 1=第二个摄像头）
CONFIDENCE_THRESHOLD = 0.5     # 置信度阈值
WINDOW_NAME = "YOLOv8 Realtime Detection (press 'q' to quit)"


# ============================================================================
# 主流程
# ============================================================================

def main():
    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 打开摄像头
    # cv2.VideoCapture 类比 Java 的 InputStream —— 打开一个数据流
    print(f"打开摄像头 (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("错误: 无法打开摄像头！请检查摄像头是否连接。")
        sys.exit(1)

    print("摄像头已打开，开始实时检测...")
    print("按 'q' 键退出\n")

    frame_count = 0

    try:
        while True:
            # 读取一帧
            # 类比 Java: byte[] frame = inputStream.read();
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面，退出。")
                break

            frame_count += 1

            # ★ 对当前帧进行检测
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,  # 不打印每帧的检测日志（否则太多）
            )

            # 在画面上绘制检测框
            # results[0].plot() 返回带标注的图片（numpy 数组）
            # 类比 Java: BufferedImage annotated = drawBoxes(frame, detections);
            annotated_frame = results[0].plot()

            # 显示画面
            cv2.imshow(WINDOW_NAME, annotated_frame)

            # 等待按键（1ms），按 'q' 退出
            # cv2.waitKey(1) 类比 Java Swing 的事件循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户按下 'q'，退出。")
                break

    finally:
        # 释放资源 — 类比 Java 的 try-finally { stream.close(); }
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n实时检测结束，共处理 {frame_count} 帧。")

    # ===================================================================
    # 【知识点：为什么 YOLO 能"实时"检测？】
    # ===================================================================
    #
    # 传统目标检测（如 R-CNN 系列）：
    #   1. 先生成候选区域（Region Proposals）→ 类似 SQL 的全表扫描
    #   2. 对每个候选区域做分类 → N 次推理
    #   3. 速度慢：每张图可能需要几秒
    #
    # YOLO 的创新：
    #   1. 将图片划分为网格（Grid）
    #   2. 一次前向传播同时预测所有网格的物体 → 类似索引查询
    #   3. 一张图只需要一次推理 → You Only Look Once!
    #   4. 速度快：RTX 4070 上 yolov8n 可达 200+ FPS
    #
    # 类比 Java 性能优化：
    #   R-CNN = for 循环逐个处理（O(N) 次推理）
    #   YOLO  = 批量处理（O(1) 次推理）
    # ===================================================================


if __name__ == "__main__":
    main()
