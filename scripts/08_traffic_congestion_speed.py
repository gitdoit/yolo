"""
08_traffic_congestion_speed.py — 基于车速估计的交通拥堵检测（方案 C）

使用 YOLO26 跟踪车辆，通过逐帧位移估算行驶速度，
用平均车速判定拥堵等级（车速越低越拥堵）。

【核心思路】
    1. YOLO26 + BoT-SORT 跟踪：每辆车分配持续 ID
    2. 记录每辆车跨帧的中心点位移（像素/帧 → 像素/秒）
    3. 统计 ROI 内所有车辆的平均速度
    4. 滑动窗口平滑 → 速度低于阈值 = 拥堵

【与方案 B 的区别】
    方案 B：面积占比 → 停满车但没人开也判拥堵
    方案 C：车速 → 更贴近真实拥堵体验（堵不堵看车速）

【运行方式】
    conda activate yolov8
    python scripts/08_traffic_congestion_speed.py

【依赖】
    ultralytics >= 8.4.0（支持 YOLO26）
"""

import time
from collections import defaultdict, deque
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "traffic_congestion_speed"

# 视频流地址
STREAM_URL = (
    "http://10.100.121.12:8086/rtp/"
    "34012200001180230000_34012200001310230002.live.flv"
)

# 断线重连
RECONNECT_DELAY = 3
MAX_RECONNECT_ATTEMPTS = 10

# 检测参数
CONFIDENCE_THRESHOLD = 0.25     # 降低阈值以检出远处车辆
IOU_THRESHOLD = 0.45

# COCO 车辆类别
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---- 速度估算参数 ----
# 保留每辆车最近 N 帧的中心点用于计算速度
TRACK_HISTORY_LEN = 30

# 每隔多少帧计算一次速度（等于检测间隔）
DETECT_EVERY_N_FRAMES = 1

# ---- 拥堵判定阈值（归一化速度）----
# 启用透视补偿后，速度 = 像素位移 / 检测框对角线 × Y坐标系数
# 无量纲，不随远近变化，需根据场景微调
# 关闭透视补偿时需改回 px/s 阈值，参考: severe=15, moderate=25, light=50
SPEED_THRESHOLDS = {
    "severe":   0.03,  # 归一化速度 < 0.03 → 严重拥堵（几乎不动）
    "moderate": 0.08,  # 0.03~0.08 → 中度拥堵（缓慢移动）
    "light":    0.15,  # 0.08~0.15 → 轻度拥堵（较慢移动）
                       # > 0.15    → 畅通
}

# 滑动窗口（用最近 N 帧的平均速度平滑判定）
SMOOTHING_WINDOW = 20

# ---- 透视补偿（方案 1 + 方案 2 组合）----
# 方案 1: Y 坐标加权 — 画面上方（远处）车辆速度乘以更大系数
# 方案 2: 检测框归一化 — 速度 / 检测框对角线，消除远近大小差异
# 启用后速度从 "像素/秒" 变为无量纲归一化值
ENABLE_PERSPECTIVE_COMP = True
Y_COMP_MAX_FACTOR = 2.0       # 画面最顶部的补偿倍数（底部恒为 1.0）

# 车辆静止判定
STATIONARY_THRESHOLD = 0.02   # 归一化速度（关闭透视补偿时改回 3.0 px/s）
STATIONARY_IGNORE = False     # True=排除停车车辆计算平均速度

# ---- 预处理参数 ----
ENABLE_PREPROCESS = True        # 总开关
ENABLE_CLAHE = True             # CLAHE 自适应直方图均衡（改善光照不均/逆光/阴影）
CLAHE_CLIP_LIMIT = 2.0          # 对比度限制（越大增强越强，2.0~4.0 常用）
CLAHE_GRID_SIZE = (8, 8)        # 分块大小
ENABLE_SHARPEN = True           # 锐化（改善模糊/压缩伪影，提升边缘清晰度）
SHARPEN_STRENGTH = 0.3          # 锐化强度（0~1，0.3 适中，太大会放大噪点）

# ---- 夜间增强模式 ----
# 解决夜间模型检测率下降的问题:
#   1. 自动判断白天/黑夜（基于画面平均亮度）
#   2. 夜间自动开启 Gamma 校正 + 增强 CLAHE + 降低置信度阈值
ENABLE_NIGHT_MODE = True            # 夜间增强总开关
NIGHT_AUTO_DETECT = True            # True=自动检测; False=手动强制夜间模式
NIGHT_BRIGHTNESS_THRESHOLD = 60     # 平均亮度低于此值判定为夜间 (0~255)
NIGHT_GAMMA = 0.4                   # Gamma 校正值 (<1 提亮暗部)
NIGHT_CLAHE_CLIP_LIMIT = 4.0        # 夜间 CLAHE 对比度限制
NIGHT_CONF_REDUCTION = 0.10         # 夜间降低置信度阈值的幅度
NIGHT_DENOISE = False               # 夜间降噪（fastNlMeans，有效但慢）
NIGHT_DENOISE_STRENGTH = 10         # 降噪强度

# ---- 显示窗口 ----
# 显示宽度（像素），高度自动按比例缩放。设为 None 则使用原始分辨率
DISPLAY_WIDTH = 1280

# 中文字体路径（Windows 默认微软雅黑）
CHINESE_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


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
# 帧预处理
# ============================================================================

def preprocess_frame(frame, is_night=False):
    """
    对输入帧做预处理以提升检测精度。
    白天: CLAHE + 锐化
    夜间: Gamma校正 + 降噪(可选) + 增强CLAHE + 锐化
    """
    if not ENABLE_PREPROCESS:
        return frame

    result = frame

    # ---- 夜间增强 ----
    if is_night and ENABLE_NIGHT_MODE:
        # Gamma 校正 — 把暗部拉亮
        inv_gamma = 1.0 / max(NIGHT_GAMMA, 0.01)
        table = np.array([
            np.clip(((i / 255.0) ** inv_gamma) * 255, 0, 255)
            for i in range(256)
        ]).astype("uint8")
        result = cv2.LUT(result, table)

        # 降噪
        if NIGHT_DENOISE:
            result = cv2.fastNlMeansDenoisingColored(
                result, None,
                h=NIGHT_DENOISE_STRENGTH,
                hForColorComponents=NIGHT_DENOISE_STRENGTH,
                templateWindowSize=7,
                searchWindowSize=21)

        # 增强CLAHE
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=NIGHT_CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        if ENABLE_SHARPEN:
            blurred = cv2.GaussianBlur(result, (0, 0), 3)
            result = cv2.addWeighted(
                result, 1.0 + SHARPEN_STRENGTH, blurred,
                -SHARPEN_STRENGTH, 0)
        return result

    # ---- 白天正常流程 ----

    # CLAHE — 在 LAB 色彩空间的 L 通道做均衡，不影响颜色
    if ENABLE_CLAHE:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 锐化 — 用 Unsharp Mask 方法
    if ENABLE_SHARPEN:
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(
            result, 1.0 + SHARPEN_STRENGTH, blurred, -SHARPEN_STRENGTH, 0)

    return result


# ============================================================================
# 速度计算
# ============================================================================

def compute_speed(track_history, track_id, dt,
                  bbox=None, frame_height=None):
    """
    计算车辆瞬时速度。
    启用透视补偿时，先除以检测框对角线（方案2），再乘 Y 坐标系数（方案1）。
    返回 (compensated_speed, raw_pixel_speed)。
    """
    pts = track_history.get(track_id)
    if pts is None or len(pts) < 2:
        return 0.0, 0.0
    p1 = pts[-2]
    p2 = pts[-1]
    dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    raw_speed = dist / max(dt, 1e-6)

    if not ENABLE_PERSPECTIVE_COMP or bbox is None or frame_height is None:
        return raw_speed, raw_speed

    x1, y1, x2, y2 = bbox

    # 方案 2: 检测框对角线归一化 → "车身/秒"
    box_diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    compensated = raw_speed / max(box_diag, 1.0)

    # 方案 1: Y 坐标加权 — 远处（画面上方）额外放大
    cy = (y1 + y2) / 2.0
    y_ratio = np.clip(cy / max(frame_height, 1), 0.05, 1.0)
    # 底部 y_ratio≈1 → factor=1.0; 顶部 y_ratio→0 → factor=Y_COMP_MAX_FACTOR
    y_factor = 1.0 + (Y_COMP_MAX_FACTOR - 1.0) * (1.0 - y_ratio)
    compensated *= y_factor

    return compensated, raw_speed


# ============================================================================
# 拥堵等级判定（基于平均速度）
# ============================================================================

def get_congestion_level(avg_speed, vehicle_count):
    """根据平均车速返回拥堵等级"""
    # 没有车辆时不是拥堵，是畅通
    if vehicle_count == 0:
        return "畅通", (0, 200, 0)
    if avg_speed < SPEED_THRESHOLDS["severe"]:
        return "严重拥堵", (0, 0, 255)
    elif avg_speed < SPEED_THRESHOLDS["moderate"]:
        return "中度拥堵", (0, 100, 255)
    elif avg_speed < SPEED_THRESHOLDS["light"]:
        return "轻度拥堵", (0, 200, 255)
    else:
        return "畅通", (0, 200, 0)


# ============================================================================
# 中文文字渲染（PIL）
# ============================================================================

def _load_font(size=20):
    """加载中文字体，加载失败则返回默认字体"""
    try:
        return ImageFont.truetype(CHINESE_FONT_PATH, size)
    except (OSError, IOError):
        return ImageFont.load_default()


def put_chinese_text(img, text, pos, color=(255, 255, 255), font_size=20):
    """用 PIL 在 OpenCV 图像上绘制中文文字"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(font_size)
    # OpenCV BGR → PIL RGB
    r, g, b = color[2], color[1], color[0]
    draw.text(pos, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================================
# 可视化
# ============================================================================

def draw_overlay(frame, tracked_vehicles, track_history,
                 avg_speed, smoothed_speed, level, color,
                 vehicle_count, fps):
    """在画面上绘制跟踪轨迹、速度和拥堵信息"""

    for v in tracked_vehicles:
        x1, y1, x2, y2 = v["bbox"]
        tid = v["track_id"]
        spd = v["speed"]

        # 检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

        # 标签：类别 + 速度
        raw = v.get("raw_speed", spd)
        label = f'{v["class"]} ID:{tid} {spd:.2f} ({raw:.0f}px/s)'
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame, (x1, y1 - th - 6), (x1 + tw, y1), (255, 200, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 轨迹线
        pts = track_history.get(tid)
        if pts and len(pts) > 1:
            points = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, (0, 255, 255), 2)

    # 信息面板（用 PIL 渲染中文）
    panel_h = 220
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (400, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)

    _pc = ENABLE_PERSPECTIVE_COMP
    lines = [
        (f"拥堵状态: {level}", color),
        (f"跟踪车辆: {vehicle_count}", (255, 255, 255)),
        (f"{'补偿' if _pc else '平均'}车速(当前): {avg_speed:.3f}", (255, 255, 255)),
        (f"{'补偿' if _pc else '平均'}车速(平滑): {smoothed_speed:.3f}", (255, 255, 255)),
        (f"FPS: {fps:.1f}", (255, 255, 255)),
        (f"模型: YOLO26n | 透视补偿: {'ON' if _pc else 'OFF'}", (180, 180, 180)),
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

    stream_fps = cap.get(cv2.CAP_PROP_FPS)
    if stream_fps <= 0:
        stream_fps = 25.0

    # 状态变量
    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LEN))
    speed_history = deque(maxlen=SMOOTHING_WINDOW)
    frame_idx = 0
    stats = {"畅通": 0, "轻度拥堵": 0, "中度拥堵": 0, "严重拥堵": 0}
    fail_count = 0
    reconn_count = 0
    prev_time = time.time()

    # 创建可调整大小的窗口
    win_name = "Speed-based Congestion - YOLO26 - 'q' to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if DISPLAY_WIDTH:
        cv2.resizeWindow(win_name, DISPLAY_WIDTH,
                         int(DISPLAY_WIDTH * 9 / 16))

    print("开始基于车速的拥堵检测... (按 'q' 退出)")
    print(f"模型: YOLO26n | 视频FPS: {stream_fps:.1f}")
    print("-" * 60)

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
        frame_idx += 1
        cur_time = time.time()
        dt = cur_time - prev_time
        prev_time = cur_time

        # 跳帧
        if frame_idx % DETECT_EVERY_N_FRAMES != 0:
            continue

        t0 = time.time()

        # 夜间自动检测 + 动态置信度调整
        is_night = False
        if ENABLE_NIGHT_MODE and NIGHT_AUTO_DETECT:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_night = np.mean(gray) < NIGHT_BRIGHTNESS_THRESHOLD
        elif ENABLE_NIGHT_MODE and not NIGHT_AUTO_DETECT:
            is_night = True

        cur_conf = CONFIDENCE_THRESHOLD
        if is_night:
            cur_conf = max(0.10, CONFIDENCE_THRESHOLD - NIGHT_CONF_REDUCTION)

        # 预处理
        processed = preprocess_frame(frame, is_night=is_night)

        # YOLO26 跟踪（persist=True 保持跨帧 ID）
        results = model.track(
            source=processed,
            conf=cur_conf,
            iou=IOU_THRESHOLD,
            imgsz=1280,
            classes=VEHICLE_CLASS_IDS,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
        )

        tracked_vehicles = []
        speeds = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                if class_id not in VEHICLE_CLASS_NAMES:
                    continue

                track_id = int(boxes.id[i].item())
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 更新轨迹
                track_history[track_id].append((cx, cy))

                # 计算速度（透视补偿）
                spd, raw_spd = compute_speed(
                    track_history, track_id, dt,
                    bbox=(x1, y1, x2, y2),
                    frame_height=frame.shape[0])

                # 是否排除静止车辆
                if STATIONARY_IGNORE and spd < STATIONARY_THRESHOLD:
                    pass
                else:
                    speeds.append(spd)

                tracked_vehicles.append({
                    "class": VEHICLE_CLASS_NAMES[class_id],
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "speed": spd,
                    "raw_speed": raw_spd,
                    "confidence": boxes.conf[i].item(),
                })

        # 清理已离开画面的轨迹
        active_ids = {v["track_id"] for v in tracked_vehicles}
        stale = [k for k in track_history if k not in active_ids]
        for k in stale:
            del track_history[k]

        # 计算平均速度
        avg_speed = np.mean(speeds) if speeds else 0.0
        speed_history.append(avg_speed)
        smoothed = np.mean(speed_history)
        level, color = get_congestion_level(smoothed, len(tracked_vehicles))
        stats[level] += 1

        fps = 1.0 / max(time.time() - t0, 1e-6)

        vis = draw_overlay(processed, tracked_vehicles, track_history,
                           avg_speed, smoothed, level, color,
                           len(tracked_vehicles), fps)

        # 高质量缩放到显示尺寸（INTER_AREA 缩小时最清晰）
        if DISPLAY_WIDTH and vis.shape[1] != DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / vis.shape[1]
            display_h = int(vis.shape[0] * scale)
            vis = cv2.resize(vis, (DISPLAY_WIDTH, display_h),
                             interpolation=cv2.INTER_AREA)

        if frame_idx % 30 == 0 or frame_idx == 1:
            print(f"[帧 {frame_idx:>6d}] 车辆: {len(tracked_vehicles)} | "
                  f"补偿速度: {avg_speed:.3f} (smooth: {smoothed:.3f}) | "
                  f"状态: {level} | FPS: {fps:.1f}")

        cv2.imshow(win_name, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n用户退出")
            break

    cap.release()
    cv2.destroyAllWindows()

    total = sum(stats.values())
    if total > 0:
        print("\n" + "=" * 60)
        print("  检测完成 — 统计摘要 (方案C: 基于车速)")
        print("=" * 60)
        print(f"  处理帧数: {total}")
        for lv, cnt in stats.items():
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {lv}: {cnt} 帧 ({pct:.1f}%) {bar}")
        dominant = max(stats, key=stats.get)
        print(f"\n  综合判定: {dominant}")
        print("=" * 60)


if __name__ == "__main__":
    main()
