"""
=============================================================================
18_vlm_flood_verify.py — YOLO + VLM 级联积水二次验证
=============================================================================

流程：
  1. YOLO 分割模型初筛 → 检出积水区域
  2. 调用本地 Ollama (qwen2.5vl:3b) 对标注图做二次判断
  3. 输出最终结论（过滤误报）

【前置条件】
  - ollama 已安装并运行：ollama serve
  - 模型已拉取：ollama pull qwen2.5vl:3b

【运行方式】
  conda activate yolov8
  python scripts/18_vlm_flood_verify.py
"""

from pathlib import Path
import base64
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vlm_flood_verify"
CONFIDENCE_THRESHOLD = 0.25
FORCE_VLM_REVIEW = True

OLLAMA_URL = "http://localhost:11434/api/chat"
VLM_MODEL = "qwen2.5vl:3b"

# 测试图片列表
TEST_IMAGES = [
    Path(r"D:\workspace\company\vision-repo\data\files\test\2026-03-31\9d01e65d-e9d3-4885-985e-2050635a95ee.jpg"),
]

# 内涝等级阈值（基于积水面积占比 %）
LEVEL_THRESHOLDS = {"light": 5, "moderate": 15, "severe": 35}

VLM_PROMPT = """你是一个专业的图像分析助手，专门用于判断监控摄像头图像中是否存在真实积水。

图中红色/彩色遮罩区域是 YOLO 模型检测到的疑似积水区域。

请判断：
1. 这些标记区域是否是真实的路面积水（而非反光、阴影、湿润路面、深色地面）？
2. 如果是真实积水，严重程度如何？

请严格按以下 JSON 格式回答（不要输出其他内容）：
{"is_flood": true或false, "confidence": 0到1之间的小数, "reason": "简短中文理由", "severity": "none或light或moderate或severe"}"""


def build_ollama_session() -> requests.Session:
    """访问本地 Ollama 时绕过系统代理，避免 localhost 请求被代理转发。"""
    session = requests.Session()
    session.trust_env = False
    return session


OLLAMA_SESSION = build_ollama_session()
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"


# ============================================================================
# 工具函数
# ============================================================================


def ensure_ollama_running():
    """检查 ollama 是否在运行，若没有则启动"""
    try:
        resp = OLLAMA_SESSION.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if resp.status_code == 200:
            print("  ollama 服务已在运行")
            return True
    except Exception:
        pass

    print("  ollama 未运行，正在启动...")
    ollama_executable = shutil.which("ollama")
    if ollama_executable is None:
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            ollama_executable = str(candidate)
    if ollama_executable is None:
        print("  ⚠ 未找到 ollama 可执行文件，请确认已安装并加入 PATH")
        return False

    subprocess.Popen(
        [ollama_executable, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import time
    for _ in range(15):
        time.sleep(1)
        try:
            resp = OLLAMA_SESSION.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
            if resp.status_code == 200:
                print("  ollama 服务已启动")
                return True
        except Exception:
            pass
    print("  ⚠ ollama 服务启动超时，请手动运行: ollama serve")
    return False


def img_to_base64(img_bgr: np.ndarray) -> str:
    """BGR 图转 base64 字符串"""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def vlm_verify(annotated_img: np.ndarray) -> dict:
    """
    把 YOLO 标注后的图发给 VLM，询问是否真实积水。
    返回 {"is_flood": bool, "confidence": float, "reason": str, "severity": str}
    """
    b64 = img_to_base64(annotated_img)

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": VLM_PROMPT,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        resp = OLLAMA_SESSION.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()

        # 尝试解析 JSON，兼容模型可能在 JSON 前后输出多余文字
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        else:
            return {"is_flood": None, "confidence": 0, "reason": content, "severity": "unknown"}

    except requests.exceptions.Timeout:
        return {"is_flood": None, "confidence": 0, "reason": "VLM 推理超时", "severity": "unknown"}
    except Exception as e:
        return {"is_flood": None, "confidence": 0, "reason": str(e), "severity": "unknown"}


def classify_level(num_detections: int, water_coverage: float) -> str:
    if num_detections == 0:
        return "无积水"
    if water_coverage < LEVEL_THRESHOLDS["light"]:
        return "微量积水"
    if water_coverage < LEVEL_THRESHOLDS["moderate"]:
        return "轻度积水"
    if water_coverage < LEVEL_THRESHOLDS["severe"]:
        return "中度内涝"
    return "严重内涝"


# ============================================================================
# 核心处理
# ============================================================================


def process_image(model: YOLO, img_path: Path, output_dir: Path):
    print(f"\n{'─' * 60}")
    print(f"图片: {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print("  ⚠ 无法读取图片，跳过")
        return None

    h, w = img.shape[:2]
    print(f"  尺寸: {w} x {h}")

    # ── 第一阶段：YOLO 推理 ────────────────────────────────────────────────
    results = model.predict(
        source=str(img_path),
        conf=CONFIDENCE_THRESHOLD,
        iou=0.5,
        verbose=False,
    )
    result = results[0]
    masks = result.masks
    boxes = result.boxes

    water_pixel_count = 0
    num_detections = 0
    max_conf = 0.0

    if masks is not None and len(masks) > 0:
        num_detections = len(masks)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for i, mask in enumerate(masks.data):
            conf = float(boxes.conf[i])
            max_conf = max(max_conf, conf)
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8))
        water_pixel_count = int(combined_mask.sum())

    water_coverage = water_pixel_count / (h * w) * 100
    yolo_level = classify_level(num_detections, water_coverage)

    print(f"\n  【YOLO 初筛】")
    print(f"    区域数:    {num_detections}")
    print(f"    最高置信度: {max_conf:.2f}")
    print(f"    积水覆盖率: {water_coverage:.1f}%")
    print(f"    初筛等级:  {yolo_level}")

    # 保存 YOLO 标注图
    annotated = result.plot()
    yolo_output = output_dir / f"yolo_{img_path.stem}.jpg"
    cv2.imwrite(str(yolo_output), annotated)

    # ── 第二阶段：VLM 复核 ───────────────────────────────────────────────
    vlm_result = {"is_flood": None, "confidence": 0, "reason": "YOLO 未检出，跳过 VLM", "severity": "none"}
    final_level = yolo_level
    is_false_positive = False

    should_run_vlm = num_detections > 0 or FORCE_VLM_REVIEW
    if should_run_vlm:
        print(f"\n  【VLM 复核】 调用 {VLM_MODEL}...")
        vlm_result = vlm_verify(annotated)

        print(f"    是否积水:  {vlm_result.get('is_flood')}")
        print(f"    VLM 置信度: {vlm_result.get('confidence', 0):.2f}")
        print(f"    理由:      {vlm_result.get('reason', '')}")
        print(f"    VLM 等级:  {vlm_result.get('severity', 'unknown')}")

        # 判断是否误报
        if vlm_result.get("is_flood") is False:
            if num_detections > 0:
                is_false_positive = True
                final_level = "误报（VLM过滤）"
        elif vlm_result.get("is_flood") is True:
            sev_map = {"none": "无积水", "light": "轻度积水", "moderate": "中度内涝", "severe": "严重内涝"}
            final_level = sev_map.get(vlm_result.get("severity", ""), yolo_level)

    print(f"\n  ✅ 最终判定: 【{final_level}】{'  ← YOLO 误报已过滤' if is_false_positive else ''}")

    # 保存最终标注图（加文字水印）
    label_img = annotated.copy()
    color = (0, 0, 255) if is_false_positive else (0, 200, 0)
    cv2.putText(label_img, f"Final: {final_level}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    final_output = output_dir / f"final_{img_path.stem}.jpg"
    cv2.imwrite(str(final_output), label_img)
    print(f"  最终标注图: {final_output}")

    return {
        "image": img_path.name,
        "yolo_detections": num_detections,
        "yolo_conf": max_conf,
        "yolo_coverage": water_coverage,
        "yolo_level": yolo_level,
        "vlm_is_flood": vlm_result.get("is_flood"),
        "vlm_confidence": vlm_result.get("confidence", 0),
        "vlm_reason": vlm_result.get("reason", ""),
        "final_level": final_level,
        "false_positive": is_false_positive,
    }


# ============================================================================
# 主函数
# ============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  YOLO + VLM 级联积水检测")
    print("=" * 65)
    print(f"强制 VLM 复核: {FORCE_VLM_REVIEW}")

    # 检查模型
    if not MODEL_PATH.exists():
        print(f"\n  ⚠ 模型不存在: {MODEL_PATH}")
        return

    # 检查 ollama
    print("\n检查 ollama 服务...")
    if not ensure_ollama_running():
        return

    print(f"\n加载 YOLO 模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 过滤有效图片
    valid_images = [p for p in TEST_IMAGES if p.exists()]
    skipped = [p for p in TEST_IMAGES if not p.exists()]
    for p in skipped:
        print(f"  ⚠ 图片不存在，已跳过: {p}")

    print(f"\n有效图片: {len(valid_images)} 张 | VLM 模型: {VLM_MODEL}")
    print(f"输出目录: {OUTPUT_DIR}")

    if not valid_images:
        print("\n  没有可测试的图片")
        return

    # 逐张处理
    summaries = []
    for p in valid_images:
        r = process_image(model, p, OUTPUT_DIR)
        if r:
            summaries.append(r)

    # 汇总
    if summaries:
        print(f"\n{'=' * 65}")
        print("  汇总")
        print(f"{'=' * 65}")
        print(f"\n{'图片':<35} {'YOLO':>6} {'VLM':>6} {'最终等级'}")
        print("─" * 65)
        for r in summaries:
            name = r["image"][:33] if len(r["image"]) > 33 else r["image"]
            yolo_ok = str(r["yolo_detections"])
            vlm_ok = "误报" if r["false_positive"] else ("✓" if r["vlm_is_flood"] else ("跳过" if r["vlm_is_flood"] is None else "无水"))
            print(f"{name:<35} {yolo_ok:>6} {vlm_ok:>6}  {r['final_level']}")

        fp_count = sum(1 for r in summaries if r["false_positive"])
        detected = sum(1 for r in summaries if r["yolo_detections"] > 0)
        print("─" * 65)
        print(f"总计: {len(summaries)} 张 | YOLO检出: {detected} 张 | VLM过滤误报: {fp_count} 张")

    print(f"\n结果已保存至: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
