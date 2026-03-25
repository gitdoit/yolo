"""
07_traffic_congestion_seg.py — 语义分割 + 手动修正 交通拥堵检测

首次运行：用 SegFormer(Cityscapes) 自动识别道路像素，用户可手动涂抹修正，
         确认后保存掩码到磁盘。
后续运行：直接加载已保存的掩码，跳过分割。

手动修正操作：
  左键拖拽 = 添加道路区域（绿色画笔）
  右键拖拽 = 擦除道路区域（红色画笔）
  滚轮     = 调整画笔大小
  R        = 重置为模型原始分割结果
  S        = 保存并开始检测
  ESC      = 不修改，直接使用当前掩码开始检测
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DETECT_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "traffic_congestion_seg"

# 语义分割模型（首次运行自动下载，约 15MB）
SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

# 视频流地址
STREAM_URL = (
    "http://10.100.121.12:8086/rtp/34020000001180001017_34020000001310000001.live.flv"
)

# 断线重连
RECONNECT_DELAY = 3
MAX_RECONNECT_ATTEMPTS = 10

# 检测参数
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# COCO 车辆类别
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Cityscapes 中道路类别 ID = 0
ROAD_CLASS_ID = 0

# 道路掩码缓存路径（保存后下次直接加载，无需重新分割）
MASK_SAVE_PATH = OUTPUT_DIR / "road_mask.png"

# 手动修正时的画笔半径（像素），运行时可用滚轮调节
BRUSH_RADIUS = 20

# 每隔多少帧做一次车辆检测
DETECT_EVERY_N_FRAMES = 2

# 拥堵阈值
CONGESTION_THRESHOLDS = {
    "free":     0.08,
    "light":    0.20,
    "moderate": 0.40,
}

SMOOTHING_WINDOW = 15


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
# 道路语义分割
# ============================================================================

def load_seg_model(model_name, device):
    """加载 SegFormer 分割模型"""
    print(f"加载道路分割模型: {model_name}")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device).eval()
    print("道路分割模型加载完成")
    return processor, model


def segment_road(frame, processor, model, device):
    """
    对一帧图像做语义分割，返回道路二值掩码 (uint8, 0/255)。
    Cityscapes label 0 = road。
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # 上采样到原图尺寸
    h, w = frame.shape[:2]
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    road_mask = np.where(pred == ROAD_CLASS_ID, 255, 0).astype(np.uint8)
    return road_mask


# ============================================================================
# 道路掩码：保存 / 加载 / 手动修正
# ============================================================================

def save_mask(mask, path):
    """保存道路掩码到磁盘"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    print(f"道路掩码已保存: {path}")


def load_mask(path):
    """从磁盘加载道路掩码，失败返回 None"""
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        print(f"已加载缓存的道路掩码: {path}")
    return mask


def edit_road_mask(frame, mask):
    """
    交互式编辑道路掩码。
    左键=添加道路  右键=擦除道路  滚轮=调整画笔  R=重置  S=保存  ESC=跳过
    返回编辑后的掩码。
    """
    edited = mask.copy()
    original = mask.copy()
    brush_r = BRUSH_RADIUS
    drawing = False
    erase_mode = False

    def _blend(frm, msk):
        vis = frm.copy()
        green = np.zeros_like(vis)
        green[:, :] = (0, 200, 0)
        vis[msk == 255] = (
            vis[msk == 255] * 0.5 + green[msk == 255] * 0.5
        ).astype(np.uint8)
        return vis

    win = "Edit Road Mask - S:save  ESC:skip  R:reset"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1280), min(frame.shape[0], 720))

    def on_mouse(event, x, y, flags, _):
        nonlocal drawing, erase_mode, edited, brush_r
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, erase_mode = True, False
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing, erase_mode = True, True
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 5 if flags > 0 else -5
            brush_r = max(5, min(100, brush_r + delta))
        if drawing:
            color = 0 if erase_mode else 255
            cv2.circle(edited, (x, y), brush_r, color, -1)

    cv2.setMouseCallback(win, on_mouse)

    print("\n" + "=" * 60)
    print("  道路掩码编辑模式")
    print("  左键拖拽=添加道路  右键拖拽=擦除  滚轮=画笔大小")
    print("  R=重置  S=保存并开始  ESC=直接开始")
    print("=" * 60)

    while True:
        vis = _blend(frame, edited)
        # 画笔指示器（右下角）
        cv2.putText(vis, f"Brush: {brush_r}px", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s') or key == ord('S'):
            save_mask(edited, MASK_SAVE_PATH)
            break
        elif key == 27:  # ESC
            print("跳过保存，使用当前掩码")
            break
        elif key == ord('r') or key == ord('R'):
            edited = original.copy()
            print("已重置为模型原始分割结果")

    cv2.destroyWindow(win)
    return edited


# ============================================================================
# 拥堵等级判定
# ============================================================================

def get_congestion_level(ratio):
    """根据车辆面积占比返回拥堵等级"""
    if ratio < CONGESTION_THRESHOLDS["free"]:
        return "畅通", (0, 200, 0)
    elif ratio < CONGESTION_THRESHOLDS["light"]:
        return "轻度拥堵", (0, 200, 255)
    elif ratio < CONGESTION_THRESHOLDS["moderate"]:
        return "中度拥堵", (0, 100, 255)
    else:
        return "严重拥堵", (0, 0, 255)


# ============================================================================
# 车辆密度计算（基于道路掩码）
# ============================================================================

def compute_vehicle_density(result, road_mask):
    """
    计算车辆在道路区域的面积占比。
    road_mask: uint8 掩码，255=道路，0=非道路
    返回: (车辆数, 面积占比, 车辆信息列表)
    """
    road_area = int(np.count_nonzero(road_mask))
    if road_area <= 0:
        return 0, 0.0, []

    vehicles = []
    total_vehicle_area = 0

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in VEHICLE_CLASS_IDS:
            continue
        confidence = box.conf[0].item()
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # 取检测框内的道路掩码区域，计算交集像素数
        box_road = road_mask[y1:y2, x1:x2]
        intersection = int(np.count_nonzero(box_road))
        box_area = (x2 - x1) * (y2 - y1)

        # 车辆框至少 30% 在道路上才计入
        if box_area > 0 and intersection >= box_area * 0.3:
            total_vehicle_area += intersection
            vehicles.append({
                "class": VEHICLE_CLASS_NAMES.get(class_id, "unknown"),
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            })

    density_ratio = total_vehicle_area / road_area
    return len(vehicles), density_ratio, vehicles


# ============================================================================
# 可视化
# ============================================================================

def draw_overlay(frame, road_mask, vehicles, vehicle_count,
                 density_ratio, avg_ratio, level, color, fps):
    """在画面上叠加道路掩码、车辆框和信息面板"""
    # 道路区域半透明着色
    overlay = frame.copy()
    overlay[road_mask == 255] = (
        overlay[road_mask == 255] * 0.6
        + np.array(color, dtype=np.float64) * 0.4
    ).astype(np.uint8)
    frame = overlay

    # 车辆检测框
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        label = f'{v["class"]} {v["confidence"]:.0%}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame, (x1, y1 - th - 6), (x1 + tw, y1), (255, 200, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 信息面板
    panel_h = 200
    cv2.rectangle(frame, (0, 0), (380, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
    road_pct = int(np.count_nonzero(road_mask)) / road_mask.size * 100

    texts = [
        (f"Congestion: {level}", color),
        (f"Vehicles on road: {vehicle_count}", (255, 255, 255)),
        (f"Density (cur): {density_ratio:.1%}", (255, 255, 255)),
        (f"Density (avg): {avg_ratio:.1%}", (255, 255, 255)),
        (f"Road coverage: {road_pct:.1f}%", (255, 255, 255)),
        (f"FPS: {fps:.1f}", (255, 255, 255)),
    ]
    for i, (text, c) in enumerate(texts):
        cv2.putText(frame, text, (10, 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
    return frame


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"推理设备: {device}")

    # 加载车辆检测模型
    detect_model = YOLO(str(DETECT_MODEL_PATH))

    # 连接视频流
    cap = open_stream(STREAM_URL)
    if cap is None:
        return

    # ---- 道路掩码：加载缓存 或 首帧分割 ----
    road_mask = load_mask(MASK_SAVE_PATH)
    if road_mask is None:
        print("未找到缓存掩码，将对首帧进行道路分割...")
        seg_processor, seg_model = load_seg_model(SEG_MODEL_NAME, device)
        ret, first_frame = cap.read()
        if not ret:
            print("错误: 无法从视频流读取首帧")
            return
        road_mask = segment_road(first_frame, seg_processor, seg_model, device)
        # 分割完成后释放分割模型，节省显存
        del seg_model, seg_processor
        torch.cuda.empty_cache() if device == "cuda" else None
        print("道路分割完成，进入手动修正模式...")
        road_mask = edit_road_mask(first_frame, road_mask)
    else:
        print(f"提示: 如需重新分割，删除 {MASK_SAVE_PATH} 后重启")

    # 状态变量
    density_history = deque(maxlen=SMOOTHING_WINDOW)
    last_detect = None
    frame_idx = 0
    stats = {"畅通": 0, "轻度拥堵": 0, "中度拥堵": 0, "严重拥堵": 0}
    fail_count = 0
    reconn_count = 0

    print("开始语义分割拥堵检测... (按 'q' 退出)")
    print("-" * 60)

    while True:
        ret, frame = cap.read()

        # 断线重连
        if not ret:
            fail_count += 1
            if fail_count > 30:
                reconn_count += 1
                if 0 < MAX_RECONNECT_ATTEMPTS < reconn_count:
                    print(f"\n已达最大重连次数，退出")
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
        frame_idx += 1
        t0 = time.time()

        # 车辆检测（道路掩码已在启动时确定，无需每帧更新）
        if frame_idx % DETECT_EVERY_N_FRAMES == 0 or last_detect is None:
            results = detect_model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=list(VEHICLE_CLASS_IDS),
                verbose=False,
            )
            last_detect = results[0]

        # 计算密度
        v_count, d_ratio, vehs = compute_vehicle_density(
            last_detect, road_mask)
        density_history.append(d_ratio)
        avg = np.mean(density_history)
        level, color = get_congestion_level(avg)
        stats[level] += 1

        fps = 1.0 / max(time.time() - t0, 1e-6)

        # 可视化
        vis = draw_overlay(frame, road_mask, vehs, v_count,
                           d_ratio, avg, level, color, fps)

        if frame_idx % 30 == 0 or frame_idx == 1:
            print(f"[帧 {frame_idx:>6d}] 车辆: {v_count} | "
                  f"占比: {d_ratio:.1%} (avg: {avg:.1%}) | "
                  f"状态: {level} | FPS: {fps:.1f}")

        cv2.imshow("Seg Traffic Detection - 'q' to quit", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n用户退出")
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_idx > 0:
        print("\n" + "=" * 60)
        print("  检测完成 — 统计摘要")
        print("=" * 60)
        print(f"  处理帧数: {frame_idx}")
        for lv, cnt in stats.items():
            pct = cnt / frame_idx * 100
            bar = "█" * int(pct / 2)
            print(f"    {lv}: {cnt} 帧 ({pct:.1f}%) {bar}")
        dominant = max(stats, key=stats.get)
        print(f"\n  综合判定: {dominant}")
        print("=" * 60)


if __name__ == "__main__":
    main()
