"""
09_builtin_speed_estimator.py — Ultralytics 内置 SpeedEstimator 体验

使用 ultralytics.solutions.SpeedEstimator 一行搞定：
    检测 + 跟踪 + 测速 + 可视化

【目的】
    对比内置方案与我们 08 脚本的效果差异，
    重点观察：透视补偿、远近车辆速度一致性、拥堵判定。

【运行方式】
    conda activate yolov8
    python scripts/09_builtin_speed_estimator.py

【依赖】
    ultralytics >= 8.4.0
"""

import time
from pathlib import Path

import cv2
from ultralytics import solutions

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n-visdrone.pt"  # 内置 SpeedEstimator 模型（已包含在 ultralytics 包中）

# 视频流地址（与 08 脚本保持一致）
STREAM_URL = (
    "http://10.100.121.12:8086/rtp/"
    "34012200001180230000_34012200001310230002.live.flv"
)

# SpeedEstimator 参数
METER_PER_PIXEL = 0.15      # 像素→米转换系数（需根据实际场景标定）
MAX_HISTORY = 5             # 历史轨迹点数
MAX_SPEED = 120             # 最大速度限制（km/h）
FPS = 25.0                  # 视频帧率（用于速度计算）

# 检测参数
CONFIDENCE = 0.25
IOU = 0.45
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# 显示
DISPLAY_WIDTH = 1280

# 断线重连
RECONNECT_DELAY = 3
MAX_RECONNECT_ATTEMPTS = 10


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
# 主流程
# ============================================================================

def main():
    cap = open_stream(STREAM_URL)
    if cap is None:
        return

    # 获取实际帧率
    stream_fps = cap.get(cv2.CAP_PROP_FPS)
    if stream_fps <= 0:
        stream_fps = FPS

    # 初始化内置 SpeedEstimator
    print(f"初始化 SpeedEstimator (模型: {MODEL_PATH.name})")
    speed_estimator = solutions.SpeedEstimator(
        model=str(MODEL_PATH),
        show=False,             # 我们自己控制显示
        meter_per_pixel=METER_PER_PIXEL,
        max_hist=MAX_HISTORY,
        fps=stream_fps,
        max_speed=MAX_SPEED,
        conf=CONFIDENCE,
        iou=IOU,
        classes=VEHICLE_CLASSES,
        verbose=False,
    )

    win_name = "Built-in SpeedEstimator - YOLO26 - 'q' to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if DISPLAY_WIDTH:
        cv2.resizeWindow(win_name, DISPLAY_WIDTH, int(DISPLAY_WIDTH * 9 / 16))

    frame_idx = 0
    fail_count = 0
    reconn_count = 0

    print("开始内置速度估计... (按 'q' 退出)")
    print(f"meter_per_pixel={METER_PER_PIXEL} | max_hist={MAX_HISTORY}")
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

        t0 = time.time()

        # 内置方案：一行搞定
        results = speed_estimator(frame)

        fps = 1.0 / max(time.time() - t0, 1e-6)

        # 获取渲染后的图像
        vis = results.plot_im

        # 缩放到显示尺寸
        if DISPLAY_WIDTH and vis.shape[1] != DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / vis.shape[1]
            display_h = int(vis.shape[0] * scale)
            vis = cv2.resize(vis, (DISPLAY_WIDTH, display_h),
                             interpolation=cv2.INTER_AREA)

        if frame_idx % 30 == 0 or frame_idx == 1:
            print(f"[帧 {frame_idx:>6d}] FPS: {fps:.1f}")

        cv2.imshow(win_name, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n用户退出")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("内置 SpeedEstimator 测试结束")


if __name__ == "__main__":
    main()
