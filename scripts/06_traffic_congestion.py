"""
=============================================================================
06_traffic_congestion.py — 实时摄像头交通拥堵检测
=============================================================================

接入实时摄像头视频流（HTTP-FLV / RTSP / RTMP），基于 YOLOv8 检测车辆密度，
自动判定交通拥堵等级。

【核心思路】
    1. 通过 OpenCV 接入实时视频流（HTTP-FLV / RTSP / RTMP）
    2. 用 YOLOv8 逐帧检测车辆（car/truck/bus/motorcycle）
    3. 在 ROI 区域内统计车辆面积占比
    4. 滑动窗口平滑 → 规则引擎判定拥堵等级

【视频流协议说明】
    ZLMediaKit 等流媒体服务器通常同时提供多种协议：
    - ws://host:port/path.live.flv   → WebSocket-FLV（浏览器用，OpenCV 不支持）
    - http://host:port/path.live.flv → HTTP-FLV（OpenCV 可直接读取 ★推荐）
    - rtsp://host:port/path          → RTSP（OpenCV 可直接读取）
    - rtmp://host:port/path          → RTMP（OpenCV 可直接读取）

    如果你拿到的是 ws:// 地址，只需把 ws:// 换成 http:// 即可。

【运行方式】
    conda activate yolov8
    python scripts/06_traffic_congestion.py

【依赖】
    无需额外安装，ultralytics + opencv 即可
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "traffic_congestion"

# ---- 视频流地址 ----
# 把你的 ws:// 地址改成 http:// 即可被 OpenCV 读取
STREAM_URL = "http://10.100.121.12:8086/rtp/34012200001180030000_34012200001310030002.live.flv"
# 如果 HTTP-FLV 不通，可以尝试 RTSP:
# STREAM_URL = "rtsp://10.100.121.12:554/rtp/34011146002008113700_34011146001328139460"

# ---- 断线重连 ----
RECONNECT_DELAY = 3         # 断线后等待几秒重连
MAX_RECONNECT_ATTEMPTS = 10  # 最大重连次数（0=无限重试）

# 检测参数
CONFIDENCE_THRESHOLD = 0.35     # 车辆检测置信度阈值（交通场景适当降低以检测远处小车）
IOU_THRESHOLD = 0.45            # NMS 的 IoU 阈值

# 只关注这些 COCO 类别（车辆相关）
# COCO 类别编号: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# 拥堵判定阈值（车辆面积占 ROI 面积的比例）
CONGESTION_THRESHOLDS = {
    "free":     0.08,   # < 8%    → 畅通
    "light":    0.20,   # 8%-20%  → 轻度拥堵
    "moderate": 0.40,   # 20%-40% → 中度拥堵
                        # > 40%   → 严重拥堵
}

# 滑动窗口大小（用最近 N 帧的平均占比来平滑判定，避免单帧抖动）
SMOOTHING_WINDOW = 15

# 每隔多少帧做一次检测（实时流建议 2~3，降低 GPU 负载）
DETECT_EVERY_N_FRAMES = 2

# ROI 区域（手动指定则跳过交互选取）
# 格式: (x, y, w, h) — 左上角坐标和宽高
# 设为 None 则首次运行时交互选取
ROI_RECT = None
# 示例: ROI_RECT = (100, 200, 600, 300)


# ============================================================================
# 视频流连接
# ============================================================================

def open_stream(url):
    """打开视频流，返回 VideoCapture 对象"""
    print(f"正在连接视频流: {url}")

    # 设置 FFmpeg 后端参数，优化实时流读取
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    # 减小缓冲区，降低延迟（实时流关键优化）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("错误: 无法连接视频流")
        print("请检查:")
        print(f"  1. 地址是否可达: {url}")
        print("  2. 是否需要换协议 (http/rtsp/rtmp)")
        print("  3. 网络是否通畅")
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"连接成功! 分辨率: {frame_w}x{frame_h}, FPS: {fps:.1f}")
    return cap


# ============================================================================
# ROI 交互选取
# ============================================================================

def select_roi(cap):
    """从视频流当前帧交互选取 ROI 区域"""
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法从视频流读取画面")
        return None

    print("\n请在弹出的窗口中用鼠标框选道路区域，然后按 Enter 确认，按 C 取消")
    roi = cv2.selectROI("选择道路区域 (ROI) - Enter确认 / C取消", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("未选择 ROI，将使用整个画面")
        h, w = frame.shape[:2]
        return (0, 0, w, h)

    print(f"已选择 ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print(f"提示: 下次可在配置区设置 ROI_RECT = {roi} 跳过选取\n")
    return roi


# ============================================================================
# 拥堵等级判定
# ============================================================================

def get_congestion_level(ratio):
    """根据车辆面积占比返回拥堵等级"""
    if ratio < CONGESTION_THRESHOLDS["free"]:
        return "畅通", (0, 200, 0)        # 绿色
    elif ratio < CONGESTION_THRESHOLDS["light"]:
        return "轻度拥堵", (0, 200, 255)  # 橙色
    elif ratio < CONGESTION_THRESHOLDS["moderate"]:
        return "中度拥堵", (0, 100, 255)  # 深橙色
    else:
        return "严重拥堵", (0, 0, 255)    # 红色


# ============================================================================
# 核心检测逻辑
# ============================================================================

def compute_vehicle_density(result, roi_rect):
    """
    计算单帧中车辆在 ROI 区域内的面积占比

    返回: (车辆数量, 面积占比, 车辆信息列表)
    """
    rx, ry, rw, rh = roi_rect
    roi_area = rw * rh
    if roi_area <= 0:
        return 0, 0.0, []

    vehicles = []
    total_vehicle_area = 0

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in VEHICLE_CLASS_IDS:
            continue

        confidence = box.conf[0].item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # 计算检测框与 ROI 的交集区域
        ix1 = max(x1, rx)
        iy1 = max(y1, ry)
        ix2 = min(x2, rx + rw)
        iy2 = min(y2, ry + rh)

        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            box_area = (x2 - x1) * (y2 - y1)

            # 只统计至少一半面积在 ROI 内的车辆
            if intersection_area >= box_area * 0.5:
                total_vehicle_area += intersection_area
                vehicles.append({
                    "class": VEHICLE_CLASS_NAMES.get(class_id, "unknown"),
                    "confidence": confidence,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

    density_ratio = total_vehicle_area / roi_area
    return len(vehicles), density_ratio, vehicles


# ============================================================================
# 可视化绘制
# ============================================================================

def draw_overlay(frame, roi_rect, vehicles, vehicle_count, density_ratio,
                 avg_ratio, congestion_level, level_color, fps):
    """在画面上绘制检测结果、ROI 区域和拥堵信息"""
    rx, ry, rw, rh = roi_rect

    # 绘制 ROI 区域
    overlay = frame.copy()
    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), level_color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), level_color, 2)

    # 绘制车辆检测框
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        label = f'{v["class"]} {v["confidence"]:.1%}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (255, 200, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 信息面板（左上角）
    panel_h = 180
    cv2.rectangle(frame, (0, 0), (340, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)

    y_offset = 28
    line_height = 28

    texts = [
        f"Congestion: {congestion_level}",
        f"Vehicles in ROI: {vehicle_count}",
        f"Density (current): {density_ratio:.1%}",
        f"Density (avg {SMOOTHING_WINDOW}f): {avg_ratio:.1%}",
        f"FPS: {fps:.1f}",
    ]

    # 拥堵等级用对应颜色，其余白色
    for i, text in enumerate(texts):
        color = level_color if i == 0 else (255, 255, 255)
        cv2.putText(frame, text, (10, y_offset + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return frame


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 加载模型 ----
    print(f"加载模型: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))

    # ---- 连接视频流 ----
    cap = open_stream(STREAM_URL)
    if cap is None:
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- 选取 ROI ----
    roi = ROI_RECT
    if roi is None:
        roi = select_roi(cap)
        if roi is None:
            cap.release()
            return

    print(f"ROI 区域: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print(f"拥堵阈值: 畅通<{CONGESTION_THRESHOLDS['free']:.0%}, "
          f"轻度<{CONGESTION_THRESHOLDS['light']:.0%}, "
          f"中度<{CONGESTION_THRESHOLDS['moderate']:.0%}, "
          f"严重>={CONGESTION_THRESHOLDS['moderate']:.0%}")
    print()

    # ---- 滑动窗口 ----
    density_history = deque(maxlen=SMOOTHING_WINDOW)

    # ---- 统计信息 ----
    frame_idx = 0
    last_result_cache = None
    congestion_stats = {"畅通": 0, "轻度拥堵": 0, "中度拥堵": 0, "严重拥堵": 0}
    consecutive_failures = 0
    reconnect_count = 0

    print("开始实时拥堵检测... (按 'q' 退出)")
    print("-" * 60)

    while True:
        ret, frame = cap.read()

        # ---- 断线重连 ----
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:  # 连续 30 帧读取失败，触发重连
                reconnect_count += 1
                if MAX_RECONNECT_ATTEMPTS > 0 and reconnect_count > MAX_RECONNECT_ATTEMPTS:
                    print(f"\n已达最大重连次数 ({MAX_RECONNECT_ATTEMPTS})，退出")
                    break

                print(f"\n视频流断开，{RECONNECT_DELAY}秒后尝试重连 "
                      f"(第 {reconnect_count} 次)...")
                cap.release()
                time.sleep(RECONNECT_DELAY)

                cap = open_stream(STREAM_URL)
                if cap is None:
                    print("重连失败，继续重试...")
                    continue

                consecutive_failures = 0
                print("重连成功，继续检测\n")
            continue

        consecutive_failures = 0
        frame_idx += 1
        t_start = time.time()

        # ---- 是否做检测 ----
        if frame_idx % DETECT_EVERY_N_FRAMES == 0 or last_result_cache is None:
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=list(VEHICLE_CLASS_IDS),
                verbose=False,
            )
            last_result_cache = results[0]

        result = last_result_cache

        # ---- 计算密度 ----
        vehicle_count, density_ratio, vehicles = compute_vehicle_density(result, roi)
        density_history.append(density_ratio)
        avg_ratio = np.mean(density_history)

        # ---- 判定拥堵等级（使用滑动平均值，更稳定） ----
        congestion_level, level_color = get_congestion_level(avg_ratio)
        congestion_stats[congestion_level] += 1

        # ---- 计算 FPS ----
        elapsed = time.time() - t_start
        fps = 1.0 / elapsed if elapsed > 0 else 0

        # ---- 绘制可视化 ----
        annotated = draw_overlay(
            frame, roi, vehicles, vehicle_count, density_ratio,
            avg_ratio, congestion_level, level_color, fps
        )

        # ---- 控制台输出（每 30 帧打印一次） ----
        if frame_idx % 30 == 0 or frame_idx == 1:
            print(f"[帧 {frame_idx:>6d}] "
                  f"车辆: {vehicle_count} | "
                  f"占比: {density_ratio:.1%} (avg: {avg_ratio:.1%}) | "
                  f"状态: {congestion_level} | "
                  f"FPS: {fps:.1f}")

        # ---- 显示 ----
        cv2.imshow("Traffic Congestion Detection - Press 'q' to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n用户手动退出")
            break

    # ---- 清理资源 ----
    cap.release()
    cv2.destroyAllWindows()

    # ---- 输出统计摘要 ----
    if frame_idx > 0:
        print("\n" + "=" * 60)
        print("  检测完成 — 统计摘要")
        print("=" * 60)
        print(f"  流地址: {STREAM_URL}")
        print(f"  处理帧数: {frame_idx}")
        print(f"  各等级帧数分布:")
        for level, count in congestion_stats.items():
            pct = count / frame_idx * 100
            bar = "█" * int(pct / 2)
            print(f"    {level:　<6s}: {count:>5d} 帧 ({pct:5.1f}%) {bar}")

        dominant_level = max(congestion_stats, key=congestion_stats.get)
        print(f"\n  综合判定: {dominant_level}")
        print("=" * 60)


if __name__ == "__main__":
    main()
