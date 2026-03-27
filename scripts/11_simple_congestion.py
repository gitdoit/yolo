"""
=============================================================================
11_simple_congestion.py — 简化版交通拥堵检测（手动画区域 + 定时采样）
=============================================================================

【核心思路】
    1. 连接实时视频流，首帧弹出窗口让用户手动框选 ROI（感兴趣区域）
    2. 每 2 秒取一帧，用 YOLO26n 做物体检测
    3. 计算 ROI 内车辆检测框面积占 ROI 面积的比例
    4. 根据面积占比判定拥堵等级（畅通/轻度/中度/严重）

【运行方式】
    conda activate yolov8
    python scripts/11_simple_congestion.py

【依赖】
    ultralytics >= 8.4.0
"""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "simple_congestion"

# ---- 视频流地址 ----
STREAM_URL = (
    "http://10.100.121.12:8086/rtp/"
    "34012200001180230000_34012200001310230002.live.flv"
)

# 断线重连
RECONNECT_DELAY = 3
MAX_RECONNECT_ATTEMPTS = 10

# ---- 检测参数 ----
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
# COCO 车辆类别: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---- 采样间隔（秒）----
SAMPLE_INTERVAL = 2

# ---- 拥堵阈值（车辆面积占 ROI 面积的比例）----
CONGESTION_THRESHOLDS = {
    "light":    0.08,   # < 8%  畅通, 8%-20% 轻度
    "moderate": 0.20,   # 20%-40% 中度
    "severe":   0.40,   # > 40% 严重
}

# ---- 显示 ----
DISPLAY_WIDTH = 1280
CHINESE_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"

# ---- ROI（手动指定则跳过交互选取）----
# 格式: (x, y, w, h)，设为 None 则首次运行时交互框选
ROI_RECT = None


# ============================================================================
# 视频流连接
# ============================================================================

def open_stream(url):
    """打开视频流"""
    print(f"正在连接视频流: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("错误: 无法连接视频流")
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"连接成功! 分辨率: {w}x{h}, FPS: {fps:.1f}")
    return cap


# ============================================================================
# ROI 交互选取
# ============================================================================

def select_roi(cap):
    """从视频流首帧交互选取 ROI 区域"""
    ret, frame = cap.read()
    if not ret:
        print("错误: 无法从视频流读取画面")
        return None

    print("\n请在弹出窗口中用鼠标框选道路区域，然后按 Enter 确认，按 C 取消")
    roi = cv2.selectROI(
        "框选道路区域 (Enter确认 / C取消)", frame, showCrosshair=True)
    cv2.destroyWindow("框选道路区域 (Enter确认 / C取消)")

    if roi == (0, 0, 0, 0):
        print("未选择 ROI，将使用整个画面")
        h, w = frame.shape[:2]
        return (0, 0, w, h)

    print(f"已选择 ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print(f"提示: 下次可在配置区设置 ROI_RECT = {roi} 跳过框选\n")
    return roi


# ============================================================================
# 拥堵判定
# ============================================================================

LEVEL_COLORS = {
    "畅通":     (0, 200, 0),
    "轻度拥堵": (0, 200, 255),
    "中度拥堵": (0, 100, 255),
    "严重拥堵": (0, 0, 255),
}


def get_congestion_level(ratio):
    """根据面积占比返回拥堵等级"""
    if ratio < CONGESTION_THRESHOLDS["light"]:
        return "畅通"
    elif ratio < CONGESTION_THRESHOLDS["moderate"]:
        return "轻度拥堵"
    elif ratio < CONGESTION_THRESHOLDS["severe"]:
        return "中度拥堵"
    else:
        return "严重拥堵"


# ============================================================================
# 面积占比计算
# ============================================================================

def compute_area_ratio(boxes, roi):
    """
    计算 ROI 内车辆检测框面积占 ROI 面积的比例。
    只统计与 ROI 有交集的检测框，且只算交集部分的面积。
    """
    rx, ry, rw, rh = roi
    roi_area = rw * rh
    if roi_area <= 0:
        return 0.0, 0, []

    total_vehicle_area = 0
    vehicle_count = 0
    vehicle_list = []

    if boxes.id is None and len(boxes) == 0:
        return 0.0, 0, []

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())
        if class_id not in VEHICLE_CLASS_NAMES:
            continue

        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

        # 计算检测框与 ROI 的交集
        ix1 = max(x1, rx)
        iy1 = max(y1, ry)
        ix2 = min(x2, rx + rw)
        iy2 = min(y2, ry + rh)

        if ix1 >= ix2 or iy1 >= iy2:
            continue  # 无交集，跳过

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        total_vehicle_area += intersection_area
        vehicle_count += 1
        vehicle_list.append({
            "class": VEHICLE_CLASS_NAMES[class_id],
            "bbox": (x1, y1, x2, y2),
            "confidence": boxes.conf[i].item(),
            "intersect": (ix1, iy1, ix2, iy2),
        })

    ratio = total_vehicle_area / roi_area
    return ratio, vehicle_count, vehicle_list


# ============================================================================
# 中文文字渲染
# ============================================================================

def _load_font(size=20):
    try:
        return ImageFont.truetype(CHINESE_FONT_PATH, size)
    except (OSError, IOError):
        return ImageFont.load_default()


def put_chinese_text(img, text, pos, color=(255, 255, 255), font_size=20):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(font_size)
    r, g, b = color[2], color[1], color[0]
    draw.text(pos, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================================
# 可视化
# ============================================================================

def draw_result(frame, roi, vehicles, ratio, level, vehicle_count,
                sample_idx):
    """在画面上绘制 ROI、检测框、拥堵信息"""
    rx, ry, rw, rh = roi
    level_color = LEVEL_COLORS[level]

    # 绘制 ROI 半透明填充
    overlay = frame.copy()
    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), level_color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # ROI 边框
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), level_color, 2)

    # 绘制每辆车的检测框
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

        label = f'{v["class"]} {v["confidence"]:.0%}'
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame, (x1, y1 - th - 6), (x1 + tw, y1), (255, 200, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 信息面板
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (380, 180), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)

    lines = [
        (f"【拥堵判定】{level}", level_color),
        (f"车辆面积占比: {ratio:.1%}", (255, 255, 255)),
        (f"ROI 内车辆数: {vehicle_count}", (255, 255, 255)),
        (f"采样次数: {sample_idx}", (180, 180, 180)),
        (f"采样间隔: {SAMPLE_INTERVAL}秒 | 模型: YOLO26n", (180, 180, 180)),
    ]
    for i, (text, c) in enumerate(lines):
        frame = put_chinese_text(frame, text, (10, 10 + i * 30), c, 20)

    return frame


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))

    cap = open_stream(STREAM_URL)
    if cap is None:
        return

    # 选取 ROI
    if ROI_RECT is not None:
        roi = ROI_RECT
        print(f"使用预设 ROI: {roi}")
    else:
        roi = select_roi(cap)
        if roi is None:
            cap.release()
            return

    # 状态变量
    sample_idx = 0
    stats = {"畅通": 0, "轻度拥堵": 0, "中度拥堵": 0, "严重拥堵": 0}
    fail_count = 0
    reconn_count = 0
    last_sample_time = 0  # 上次采样时间戳

    win_name = "Simple Congestion - 'q' to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if DISPLAY_WIDTH:
        cv2.resizeWindow(win_name, DISPLAY_WIDTH,
                         int(DISPLAY_WIDTH * 9 / 16))

    print("=" * 60)
    print("  简化版交通拥堵检测")
    print(f"  模型: {MODEL_PATH.name}")
    print(f"  采样间隔: {SAMPLE_INTERVAL} 秒")
    print(f"  拥堵阈值: {CONGESTION_THRESHOLDS}")
    print(f"  ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print("  按 'q' 退出")
    print("=" * 60)

    # 保存最近一次检测结果用于显示
    last_vis = None

    while True:
        ret, frame = cap.read()

        if not ret:
            fail_count += 1
            if fail_count > 30:
                reconn_count += 1
                if 0 < MAX_RECONNECT_ATTEMPTS < reconn_count:
                    print("\n已达最大重连次数，退出")
                    break
                print(f"\n断线，{RECONNECT_DELAY}s 后重连 "
                      f"(第 {reconn_count} 次)...")
                cap.release()
                time.sleep(RECONNECT_DELAY)
                cap = open_stream(STREAM_URL)
                if cap is None:
                    continue
                fail_count = 0
            continue

        fail_count = 0
        cur_time = time.time()

        # 每 SAMPLE_INTERVAL 秒采样一次
        if cur_time - last_sample_time >= SAMPLE_INTERVAL:
            last_sample_time = cur_time
            sample_idx += 1

            # YOLO26 检测（单帧检测，不需要跟踪）
            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=1280,
                classes=VEHICLE_CLASS_IDS,
                verbose=False,
            )

            # 计算面积占比
            ratio, vehicle_count, vehicles = compute_area_ratio(
                results[0].boxes, roi)

            # 判定拥堵等级
            level = get_congestion_level(ratio)
            stats[level] += 1

            # 可视化
            vis = draw_result(
                frame.copy(), roi, vehicles, ratio, level,
                vehicle_count, sample_idx)

            # 终端输出
            print(
                f"[采样 {sample_idx:>4d}] "
                f"车辆: {vehicle_count} | "
                f"面积占比: {ratio:.1%} | "
                f"判定: {level}"
            )

            last_vis = vis

        # 非采样帧：显示最近结果或原始画面（带ROI框）
        if last_vis is None:
            # 还没采样过，显示带 ROI 的原始画面
            display = frame.copy()
            rx, ry, rw, rh = roi
            cv2.rectangle(display, (rx, ry),
                          (rx + rw, ry + rh), (0, 255, 0), 2)
            display = put_chinese_text(
                display, "等待首次采样...", (10, 10), (0, 255, 0), 24)
        else:
            display = last_vis

        # 缩放显示
        if DISPLAY_WIDTH and display.shape[1] != DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / display.shape[1]
            display_h = int(display.shape[0] * scale)
            display = cv2.resize(display, (DISPLAY_WIDTH, display_h),
                                 interpolation=cv2.INTER_AREA)

        cv2.imshow(win_name, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n用户退出")
            break

    cap.release()
    cv2.destroyAllWindows()

    # 统计摘要
    total = sum(stats.values())
    if total > 0:
        print("\n" + "=" * 60)
        print("  检测完成 — 统计摘要")
        print("=" * 60)
        print(f"  采样次数: {total}")
        print(f"  采样间隔: {SAMPLE_INTERVAL} 秒")
        for lv, cnt in stats.items():
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {lv}: {cnt} 次 ({pct:.1f}%) {bar}")
        dominant = max(stats, key=stats.get)
        print(f"\n  综合判定: {dominant}")
        print("=" * 60)


if __name__ == "__main__":
    main()
