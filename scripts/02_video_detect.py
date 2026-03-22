"""
=============================================================================
02_video_detect.py — 视频文件物体检测
=============================================================================

在掌握了图片检测之后，进阶到视频检测。
视频本质上就是"连续的图片帧"，YOLO 对每一帧做检测，然后拼起来。

【与图片检测的区别】
    图片检测: model.predict("photo.jpg") → 处理 1 帧
    视频检测: model.predict("video.mp4") → 逐帧处理，自动输出带标注的视频

【Java 工程师理解方式】
    视频检测 ≈ 批量接口调用：
    for (Frame frame : video.getFrames()) {
        List<Detection> results = service.detect(frame);
        annotatedVideo.addFrame(drawBoxes(frame, results));
    }
    YOLO 帮你把这个循环封装好了。

【运行方式】
    conda activate yolov8-py310
    python scripts/02_video_detect.py

【说明】
    - 如果 data/videos/ 没有视频文件，脚本会用内置的示例视频
    - 处理后的视频保存在 outputs/video_detect/
    - 处理速度取决于 GPU 性能和视频分辨率
"""

from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 检测参数
CONFIDENCE_THRESHOLD = 0.5

# 视频检测专属参数
SHOW_REALTIME = True   # 是否实时弹窗显示检测过程（按 q 退出）


# ============================================================================
# 主流程
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # 查找视频文件
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(VIDEOS_DIR.glob(ext))

    if not video_files:
        # 没有本地视频，使用 Ultralytics 内置的示例视频 URL
        print("data/videos/ 目录为空，使用 Ultralytics 内置示例视频...")
        print("提示: 你可以把任意 mp4/avi 视频放到 data/videos/ 目录下再运行")
        source = "https://ultralytics.com/assets/decelera_portrait_min.mov"
    else:
        source = str(video_files[0])
        print(f"使用本地视频: {source}")

    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # ===================================================================
    # ★ 视频检测 — 与图片检测几乎一样，只是 source 传视频路径
    # ===================================================================
    print("开始视频检测...")
    print("（如果弹出了预览窗口，按 'q' 键可退出）\n")

    results = model.predict(
        source=source,
        conf=CONFIDENCE_THRESHOLD,
        save=True,                  # 保存带标注的视频
        show=SHOW_REALTIME,         # 实时显示检测过程
        project=str(OUTPUT_DIR),
        name="video_detect",
        exist_ok=True,
        stream=True,                # ★ 重要: 逐帧返回结果，避免一次性加载整个视频到内存
    )

    # stream=True 时，results 是一个生成器（Generator）
    # 类比 Java: 类似 Stream<Frame>，惰性求值，用多少取多少
    frame_count = 0
    for result in results:
        frame_count += 1
        num_objects = len(result.boxes)
        if frame_count % 30 == 0:  # 每30帧打印一次（避免刷屏）
            print(f"  已处理 {frame_count} 帧, 当前帧检测到 {num_objects} 个物体")

    print(f"\n视频检测完成！共处理 {frame_count} 帧")
    print(f"带标注的视频已保存到: {OUTPUT_DIR / 'video_detect'}")

    # ===================================================================
    # 【知识点：stream 参数】
    # ===================================================================
    # stream=False (默认): 一次性处理所有帧，全部结果加载到内存
    #   → 适合短视频或需要回溯访问结果的场景
    #   → 类比 Java: List.of(results)
    #
    # stream=True: 逐帧处理，每次只返回一帧结果
    #   → 适合长视频或内存受限场景
    #   → 类比 Java: Stream.generate(() -> processNextFrame())
    #   → 注意: 每帧结果只能访问一次（生成器特性）
    # ===================================================================


if __name__ == "__main__":
    main()
