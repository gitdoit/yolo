"""
=============================================================================
10_congestion_dual_criteria.py — 双条件融合交通拥堵检测（论文启发增强版）
=============================================================================

参考论文: "A lightweight end-to-end traffic congestion detection framework
          using HRTNet on the Qinghai-Tibet plateau"
          (Scientific Reports, 2025, DOI: 10.1038/s41598-025-13550-x)

【论文核心启发】
    论文的 TCMS（交通拥堵监测系统）使用双条件判定拥堵：
    1. 车辆平均车速 < 阈值（10 km/h）
    2. 交通流量 > 道路最大设计通行容量的 80%
    满足任一条件即判定为拥堵。此外还跟踪拥堵持续时间。

【本脚本改进点 vs 08 脚本】
    ① 双条件融合: 密度 + 速度 同时参与拥堵判定（08 只用速度，06/07 只用密度）
    ② 拥堵时长累计: 连续拥堵帧数 × 帧间隔 → 实时显示"已拥堵 XX 秒"
    ③ 帧聚合密度: N 帧内平均车辆数/ROI 面积（论文 Eq.7-8），比单帧计数更稳定
    ④ 支持相机标定: 可选提供标定矩阵，将像素速度转为真实速度（km/h）
    ⑤ 去雨/去雾预处理: 保留 08 的 CLAHE+锐化，新增可选去雾增强

【与论文的区别】
    论文使用自研 HRTNet 模型做检测+去雨，我们直接用 YOLO26+预处理替代。
    论文用 SORT 跟踪，我们用 BoT-SORT（更鲁棒）。
    论文用精确标定矩阵，我们提供 "标定模式" 和 "启发式透视补偿" 两种选择。

【运行方式】
    conda activate yolov8
    python scripts/10_congestion_dual_criteria.py

【依赖】
    ultralytics >= 8.4.0
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
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n-visdrone.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "congestion_dual"

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
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---- 跟踪参数 ----
TRACK_HISTORY_LEN = 30
DETECT_EVERY_N_FRAMES = 1

# ============================================================================
# 【改进 ①】双条件融合拥堵判定参数
# ============================================================================
# 论文方法: 拥堵 = 平均车速 < V_threshold  OR  流量 > 80% 设计容量
# 我们综合使用 "速度条件" + "密度条件"，分别给出等级后取更严重的那个

# -- 速度条件阈值（与 08 脚本一致）--
SPEED_THRESHOLDS = {
    "severe":   0.03,   # 归一化速度 < 0.03 → 严重拥堵
    "moderate": 0.08,   # 0.03~0.08 → 中度拥堵
    "light":    0.15,   # 0.08~0.15 → 轻度拥堵
}

# -- 密度条件阈值（车辆数/ROI 面积归一化，与 06 脚本思路一致但用车辆数）--
# 这参考了论文 Eq.7: 单位时间平均车辆数 → 密度
# DENSITY_METRIC 决定使用哪种密度计算方式：
#   "count"    → 纯车辆数（简单，适合固定视角）
#   "area"     → 检测框面积占比（与 06 脚本一致）
DENSITY_METRIC = "area"

DENSITY_THRESHOLDS = {
    "severe":   0.40,   # 面积占比 > 40% → 严重拥堵
    "moderate": 0.20,   # 20%-40% → 中度拥堵
    "light":    0.08,   # 8%-20% → 轻度拥堵
}

# -- 双条件融合策略 --
# "max"  → 取两个条件中更严重的等级（论文方式: 满足任一即判拥堵）
# "avg"  → 取平均严重度（更保守）
# "speed_only" / "density_only" → 单条件退化模式（用于对比）
FUSION_STRATEGY = "max"

# ============================================================================
# 【改进 ②】拥堵时长累计
# ============================================================================
# 论文 Eq.11: T_congestion = Σ Δt_i（拥堵帧的帧间隔累加）
# 非畅通状态的连续帧时间累计 → 显示 "已拥堵 XX 秒"
# 恢复畅通后重置

# ============================================================================
# 【改进 ③】帧聚合密度
# ============================================================================
# 论文 Eq.7: D_t = (1/T) * Σ V_i / N
# 用滑动窗口聚合最近 N 帧的车辆数/密度

SMOOTHING_WINDOW = 20      # 速度/密度共用的平滑窗口大小
DENSITY_WINDOW = 15         # 密度专用的聚合窗口

# ============================================================================
# 【改进 ④】相机标定（可选）
# ============================================================================
# 论文 Eq.9,12: 使用标定矩阵 M 将像素坐标转为真实世界坐标
# 设为 None 则使用启发式透视补偿（与 08 脚本一致）
# 如果有标定数据，可以设置为 3x3 numpy 矩阵:
#   CALIBRATION_MATRIX = np.array([[...], [...], [...]])
CALIBRATION_MATRIX = None

# 使用标定时的帧率（用于 px/帧 → m/s 转换）
CALIBRATED_FPS = 25.0

# ---- 透视补偿（无标定时的替代方案）----
ENABLE_PERSPECTIVE_COMP = True
Y_COMP_MAX_FACTOR = 2.0

# 车辆静止判定
STATIONARY_THRESHOLD = 0.02
STATIONARY_IGNORE = False

# ---- 预处理参数 ----
ENABLE_PREPROCESS = True
ENABLE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
ENABLE_SHARPEN = True
SHARPEN_STRENGTH = 0.3

# 【改进 ⑤】去雾/去雨增强（灵感: 论文 HRTNet 内置去雨模块）
# 我们用 OpenCV 实现简易暗通道去雾（Dark Channel Prior），
# 对雨天/雾天场景有一定改善作用
ENABLE_DEHAZE = False       # 默认关闭，雨雾天气时可开启
DEHAZE_STRENGTH = 0.7       # 去雾强度 0~1（越大去雾越强，但可能过曝）

# 【改进 ⑥】夜间增强模式
# 核心问题: 夜间画面整体偏暗，YOLO 无法提取有效特征，检测率骤降
# 解决思路:
#   1. 自动判断是否为夜间（基于画面平均亮度）
#   2. 夜间自动开启 Gamma 校正 + 增强版 CLAHE + 降低置信度阈值
#   3. 可选降噪（夜间传感器噪点多，但会牺牲帧率）
ENABLE_NIGHT_MODE = True            # 夜间增强总开关
NIGHT_AUTO_DETECT = True            # True=自动检测白天/黑夜; False=手动
NIGHT_BRIGHTNESS_THRESHOLD = 60     # 画面平均亮度低于此值判定为夜间 (0~255)
NIGHT_GAMMA = 0.4                   # Gamma 校正值 (<1 提亮暗部, 0.3~0.5 适合夜间)
NIGHT_CLAHE_CLIP_LIMIT = 4.0        # 夜间 CLAHE 对比度限制（比白天更强）
NIGHT_CONF_REDUCTION = 0.10         # 夜间自动降低置信度阈值的幅度
NIGHT_DENOISE = False               # 夜间降噪（fastNlMeans，有效但慢，酌情开启）
NIGHT_DENOISE_STRENGTH = 10         # 降噪强度 (3~15, 越大越平滑但丢细节)

# ---- 显示窗口 ----
DISPLAY_WIDTH = 1280
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

def dehaze(frame, strength=0.7):
    """
    简易暗通道去雾（Dark Channel Prior, He et al. 2009）。
    灵感来源: 论文 HRTNet 将去雨模块内置于检测管道，
    我们在预处理阶段用经典算法实现类似效果。

    原理：雾天图像的暗通道值偏高，通过估算大气光和透射率来恢复清晰图像。
    对雨天也有一定效果（雨纹造成的灰蒙蒙效果类似薄雾）。
    """
    img = frame.astype(np.float64) / 255.0

    # 暗通道 — 取 RGB 三通道最小值后做最小值滤波
    dark = np.min(img, axis=2)
    kernel_size = max(15, min(frame.shape[:2]) // 40)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark = cv2.erode(dark, kernel)

    # 估算大气光 A — 取暗通道最亮 0.1% 像素对应位置的原图均值
    num_pixels = dark.size
    num_brightest = max(int(num_pixels * 0.001), 1)
    flat_dark = dark.ravel()
    indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
    atmos_light = np.mean(
        img.reshape(-1, 3)[indices], axis=0).clip(0.5, 1.0)

    # 估算透射率 t
    norm_img = img / atmos_light
    t = 1.0 - strength * np.min(norm_img, axis=2)
    t = np.clip(t, 0.1, 1.0)

    # 恢复: J = (I - A) / t + A
    t_3ch = t[:, :, np.newaxis]
    result = (img - atmos_light) / t_3ch + atmos_light
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def detect_night(frame):
    """
    判断当前帧是否为夜间场景。
    方法: 将图像转为灰度图，计算平均亮度。
    低于阈值则判定为夜间。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < NIGHT_BRIGHTNESS_THRESHOLD, mean_brightness


def gamma_correction(frame, gamma=0.4):
    """
    Gamma 校正 — 夜间增强的核心手段。

    原理: 输出 = 输入^gamma
    gamma < 1 → 暗部被大幅提亮，亮部变化小（正是夜间需要的效果）
    gamma = 0.4 时，亮度 50/255 的像素会被拉到 ~130/255

    为什么比简单调亮度好？
    - 调亮度: 所有像素加同一个值，车灯等高亮区会过曝
    - Gamma: 非线性映射，暗部提升多、亮部提升少，不容易过曝
    """
    # 构建 Gamma 查找表 (LUT)，256 个值只需计算一次
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([
        np.clip(((i / 255.0) ** inv_gamma) * 255, 0, 255)
        for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(frame, table)


def preprocess_frame(frame, is_night=False):
    """
    帧预处理管道:
        白天: 去雾(可选) → CLAHE → 锐化
        夜间: Gamma校正 → 降噪(可选) → 增强CLAHE → 锐化
    """
    if not ENABLE_PREPROCESS:
        return frame

    result = frame

    # ---- 夜间增强 ----
    if is_night and ENABLE_NIGHT_MODE:
        # 1. Gamma 校正 — 把暗部拉亮（最关键的一步）
        result = gamma_correction(result, NIGHT_GAMMA)

        # 2. 降噪（夜间传感器噪点多，降噪后特征更清晰）
        if NIGHT_DENOISE:
            result = cv2.fastNlMeansDenoisingColored(
                result, None,
                h=NIGHT_DENOISE_STRENGTH,
                hForColorComponents=NIGHT_DENOISE_STRENGTH,
                templateWindowSize=7,
                searchWindowSize=21)

        # 3. 增强版 CLAHE（clipLimit 比白天更高，拉开对比度）
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=NIGHT_CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 4. 锐化
        if ENABLE_SHARPEN:
            blurred = cv2.GaussianBlur(result, (0, 0), 3)
            result = cv2.addWeighted(
                result, 1.0 + SHARPEN_STRENGTH, blurred,
                -SHARPEN_STRENGTH, 0)
        return result

    # ---- 白天正常流程 ----
    # 去雾（适用于雨天/雾天场景）
    if ENABLE_DEHAZE:
        result = dehaze(result, DEHAZE_STRENGTH)

    # CLAHE — LAB 色彩空间 L 通道均衡
    if ENABLE_CLAHE:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 锐化 — Unsharp Mask
    if ENABLE_SHARPEN:
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(
            result, 1.0 + SHARPEN_STRENGTH, blurred, -SHARPEN_STRENGTH, 0)

    return result


# ============================================================================
# 【改进 ④】速度计算（支持标定矩阵）
# ============================================================================

def pixel_to_world(px, py, M):
    """用标定矩阵将像素坐标转为真实世界坐标（米）"""
    pt = np.array([px, py, 1.0])
    world = M @ pt
    # 齐次坐标归一化
    world /= world[2]
    return world[0], world[1]


def compute_speed(track_history, track_id, dt,
                  bbox=None, frame_height=None):
    """
    计算车辆速度。
    优先使用标定矩阵（论文 Eq.9-10），否则用透视补偿。
    返回 (compensated_speed, raw_pixel_speed)
    """
    pts = track_history.get(track_id)
    if pts is None or len(pts) < 2:
        return 0.0, 0.0

    p1 = pts[-2]
    p2 = pts[-1]

    # 有标定矩阵时，转为真实世界距离（米）
    if CALIBRATION_MATRIX is not None:
        w1x, w1y = pixel_to_world(p1[0], p1[1], CALIBRATION_MATRIX)
        w2x, w2y = pixel_to_world(p2[0], p2[1], CALIBRATION_MATRIX)
        real_dist = np.sqrt((w2x - w1x) ** 2 + (w2y - w1y) ** 2)
        # m/s → km/h
        speed_kmh = (real_dist / max(dt, 1e-6)) * 3.6
        raw_speed = np.sqrt(
            (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) / max(dt, 1e-6)
        return speed_kmh, raw_speed

    # 无标定时，走启发式透视补偿（与 08 脚本一致）
    dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    raw_speed = dist / max(dt, 1e-6)

    if not ENABLE_PERSPECTIVE_COMP or bbox is None or frame_height is None:
        return raw_speed, raw_speed

    x1, y1, x2, y2 = bbox
    box_diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    compensated = raw_speed / max(box_diag, 1.0)

    cy = (y1 + y2) / 2.0
    y_ratio = np.clip(cy / max(frame_height, 1), 0.05, 1.0)
    y_factor = 1.0 + (Y_COMP_MAX_FACTOR - 1.0) * (1.0 - y_ratio)
    compensated *= y_factor

    return compensated, raw_speed


# ============================================================================
# 【改进 ③】帧聚合密度计算
# ============================================================================

def compute_density(vehicles, frame_shape):
    """
    【论文 Eq.7-8】计算当前帧的交通密度。
    area 模式: 所有车辆检测框总面积 / 画面总面积
    count 模式: 车辆数 / 画面面积（归一化到 0~1 范围）
    """
    if not vehicles:
        return 0.0

    frame_area = frame_shape[0] * frame_shape[1]

    if DENSITY_METRIC == "area":
        total_vehicle_area = 0
        for v in vehicles:
            x1, y1, x2, y2 = v["bbox"]
            total_vehicle_area += (x2 - x1) * (y2 - y1)
        return total_vehicle_area / max(frame_area, 1)
    else:  # "count"
        return len(vehicles) / max(frame_area, 1) * 100000


# ============================================================================
# 【改进 ①】双条件融合拥堵判定
# ============================================================================

LEVEL_SEVERITY = {
    "畅通": 0,
    "轻度拥堵": 1,
    "中度拥堵": 2,
    "严重拥堵": 3,
}

SEVERITY_TO_LEVEL = {v: k for k, v in LEVEL_SEVERITY.items()}

LEVEL_COLORS = {
    "畅通":     (0, 200, 0),
    "轻度拥堵": (0, 200, 255),
    "中度拥堵": (0, 100, 255),
    "严重拥堵": (0, 0, 255),
}


def get_speed_level(avg_speed, vehicle_count):
    """基于平均车速判定拥堵等级"""
    if vehicle_count == 0:
        return "畅通"
    if avg_speed < SPEED_THRESHOLDS["severe"]:
        return "严重拥堵"
    elif avg_speed < SPEED_THRESHOLDS["moderate"]:
        return "中度拥堵"
    elif avg_speed < SPEED_THRESHOLDS["light"]:
        return "轻度拥堵"
    return "畅通"


def get_density_level(density):
    """基于交通密度判定拥堵等级"""
    if density > DENSITY_THRESHOLDS["severe"]:
        return "严重拥堵"
    elif density > DENSITY_THRESHOLDS["moderate"]:
        return "中度拥堵"
    elif density > DENSITY_THRESHOLDS["light"]:
        return "轻度拥堵"
    return "畅通"


def fuse_congestion_levels(speed_level, density_level):
    """
    【论文启发】融合两个条件的拥堵等级。
    论文: 满足 速度<阈值 OR 流量>容量 即判拥堵
    我们: 对应 "max" 策略 — 取两个条件中更严重的
    """
    if FUSION_STRATEGY == "speed_only":
        return speed_level

    if FUSION_STRATEGY == "density_only":
        return density_level

    s_sev = LEVEL_SEVERITY[speed_level]
    d_sev = LEVEL_SEVERITY[density_level]

    if FUSION_STRATEGY == "max":
        # 论文方式: 取更严重的（满足任一条件即判拥堵）
        final_sev = max(s_sev, d_sev)
    else:  # "avg"
        final_sev = round((s_sev + d_sev) / 2)

    return SEVERITY_TO_LEVEL[min(final_sev, 3)]


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

def draw_overlay(frame, tracked_vehicles, track_history,
                 avg_speed, smoothed_speed, smoothed_density,
                 speed_level, density_level, fused_level,
                 congestion_duration, vehicle_count, fps,
                 is_night=False, brightness=128.0):
    """绘制增强版信息面板"""

    fused_color = LEVEL_COLORS[fused_level]

    for v in tracked_vehicles:
        x1, y1, x2, y2 = v["bbox"]
        tid = v["track_id"]
        spd = v["speed"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

        label = f'{v["class"]} ID:{tid} {spd:.2f}'
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame, (x1, y1 - th - 6), (x1 + tw, y1), (255, 200, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        pts = track_history.get(tid)
        if pts and len(pts) > 1:
            points = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, (0, 255, 255), 2)

    # 信息面板
    panel_h = 340
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (440, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)

    # 拥堵时长文本
    if congestion_duration > 0:
        dur_text = f"已拥堵: {congestion_duration:.0f}秒"
    else:
        dur_text = "未拥堵"

    _pc = ENABLE_PERSPECTIVE_COMP
    _cal = CALIBRATION_MATRIX is not None
    speed_unit = "km/h" if _cal else ("归一化" if _pc else "px/s")
    night_text = f"夜间模式 (亮度:{brightness:.0f})" if is_night else f"白天模式 (亮度:{brightness:.0f})"

    lines = [
        (f"【融合判定】{fused_level}", fused_color),
        (f"  速度条件: {speed_level}", LEVEL_COLORS[speed_level]),
        (f"  密度条件: {density_level}", LEVEL_COLORS[density_level]),
        (f"  融合策略: {FUSION_STRATEGY}", (180, 180, 180)),
        (f"拥堵时长: {dur_text}", (255, 200, 100)),
        (f"跟踪车辆: {vehicle_count}", (255, 255, 255)),
        (f"平滑车速: {smoothed_speed:.3f} {speed_unit}", (255, 255, 255)),
        (f"平滑密度: {smoothed_density:.3f}", (255, 255, 255)),
        (f"FPS: {fps:.1f}", (255, 255, 255)),
        (f"光照: {night_text}", (100, 200, 255) if is_night else (180, 180, 180)),
        (f"模型: YOLO26n | 去雾: {'ON' if ENABLE_DEHAZE else 'OFF'}",
         (180, 180, 180)),
    ]
    for i, (text, c) in enumerate(lines):
        frame = put_chinese_text(frame, text, (10, 10 + i * 28), c, 18)
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
    density_history = deque(maxlen=DENSITY_WINDOW)
    frame_idx = 0
    stats = {"畅通": 0, "轻度拥堵": 0, "中度拥堵": 0, "严重拥堵": 0}
    fail_count = 0
    reconn_count = 0
    prev_time = time.time()

    # 【改进 ②】拥堵时长累计
    congestion_start_time = None   # 拥堵开始时间戳
    congestion_duration = 0.0      # 当前拥堵持续秒数

    win_name = "Dual-Criteria Congestion - 'q' to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if DISPLAY_WIDTH:
        cv2.resizeWindow(win_name, DISPLAY_WIDTH,
                         int(DISPLAY_WIDTH * 9 / 16))

    print("=" * 60)
    print("  双条件融合拥堵检测 (论文启发增强版)")
    print(f"  融合策略: {FUSION_STRATEGY}")
    print(f"  速度阈值: {SPEED_THRESHOLDS}")
    print(f"  密度阈值: {DENSITY_THRESHOLDS}")
    print(f"  密度指标: {DENSITY_METRIC}")
    print(f"  去雾增强: {'ON' if ENABLE_DEHAZE else 'OFF'}")
    print(f"  夜间增强: {'AUTO' if NIGHT_AUTO_DETECT else ('ON' if ENABLE_NIGHT_MODE else 'OFF')}")
    print(f"  相机标定: {'ON (真实 km/h)' if CALIBRATION_MATRIX is not None else 'OFF (透视补偿)'}")
    print("=" * 60)

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

        if frame_idx % DETECT_EVERY_N_FRAMES != 0:
            continue

        t0 = time.time()

        # 【改进 ⑥】夜间自动检测 + 动态置信度调整
        is_night = False
        brightness = 128.0
        if ENABLE_NIGHT_MODE and NIGHT_AUTO_DETECT:
            is_night, brightness = detect_night(frame)
        elif ENABLE_NIGHT_MODE and not NIGHT_AUTO_DETECT:
            is_night = True  # 手动强制夜间模式

        # 夜间自动降低置信度阈值（让暗处车辆也能被检出）
        cur_conf = CONFIDENCE_THRESHOLD
        if is_night:
            cur_conf = max(0.10, CONFIDENCE_THRESHOLD - NIGHT_CONF_REDUCTION)

        # 预处理
        processed = preprocess_frame(frame, is_night=is_night)

        # YOLO26 跟踪
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

                track_history[track_id].append((cx, cy))

                spd, raw_spd = compute_speed(
                    track_history, track_id, dt,
                    bbox=(x1, y1, x2, y2),
                    frame_height=frame.shape[0])

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

        # ---- 速度分析 ----
        avg_speed = np.mean(speeds) if speeds else 0.0
        speed_history.append(avg_speed)
        smoothed_speed = np.mean(speed_history)

        # ---- 【改进 ③】帧聚合密度分析 ----
        cur_density = compute_density(tracked_vehicles, frame.shape)
        density_history.append(cur_density)
        smoothed_density = np.mean(density_history)

        # ---- 【改进 ①】双条件融合判定 ----
        speed_level = get_speed_level(smoothed_speed, len(tracked_vehicles))
        density_level = get_density_level(smoothed_density)
        fused_level = fuse_congestion_levels(speed_level, density_level)
        stats[fused_level] += 1

        # ---- 【改进 ②】拥堵时长更新 ----
        if fused_level != "畅通":
            if congestion_start_time is None:
                congestion_start_time = cur_time
            congestion_duration = cur_time - congestion_start_time
        else:
            congestion_start_time = None
            congestion_duration = 0.0

        fps = 1.0 / max(time.time() - t0, 1e-6)

        vis = draw_overlay(
            processed, tracked_vehicles, track_history,
            avg_speed, smoothed_speed, smoothed_density,
            speed_level, density_level, fused_level,
            congestion_duration, len(tracked_vehicles), fps,
            is_night=is_night, brightness=brightness)

        if DISPLAY_WIDTH and vis.shape[1] != DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / vis.shape[1]
            display_h = int(vis.shape[0] * scale)
            vis = cv2.resize(vis, (DISPLAY_WIDTH, display_h),
                             interpolation=cv2.INTER_AREA)

        if frame_idx % 30 == 0 or frame_idx == 1:
            print(
                f"[帧 {frame_idx:>6d}] "
                f"车辆: {len(tracked_vehicles)} | "
                f"速度: {smoothed_speed:.3f} → {speed_level} | "
                f"密度: {smoothed_density:.3f} → {density_level} | "
                f"融合: {fused_level} | "
                f"拥堵时长: {congestion_duration:.0f}s | "
                f"{'[夜]' if is_night else '[昼]'} 亮度:{brightness:.0f} | "
                f"FPS: {fps:.1f}"
            )

        cv2.imshow(win_name, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n用户退出")
            break

    cap.release()
    cv2.destroyAllWindows()

    total = sum(stats.values())
    if total > 0:
        print("\n" + "=" * 60)
        print("  检测完成 — 统计摘要 (双条件融合拥堵检测)")
        print("=" * 60)
        print(f"  融合策略: {FUSION_STRATEGY}")
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
