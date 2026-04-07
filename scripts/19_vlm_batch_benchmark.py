"""
=============================================================================
19_vlm_batch_benchmark.py — VLM 批量图片性能测试
=============================================================================

用途：
  - 对本地 Ollama 视觉模型做批量图片推理性能测试
  - 默认输入 20 张图片
  - 保存中间结果，便于随时查看

保存内容：
  - results.jsonl      每张图片一条结果，边跑边追加
  - results.csv        结构化汇总表
  - summary.json       最终汇总统计
  - raw/*.json         每张图片的原始 Ollama 返回
  - prompt.txt         本次测试使用的提示词
  - images.txt         本次测试图片清单

【运行方式】
  conda activate yolov8
  python scripts/19_vlm_batch_benchmark.py
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

try:
    import psutil
except ImportError:
    psutil = None

# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "datasets" / "flood" / "split_train" / "images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "vlm_batch_benchmark"

VLM_MODEL = "qwen3-vl:2b"
MAX_IMAGES = 20
REQUEST_TIMEOUT = 180
WARMUP_FIRST = False
FORCE_CPU = True
MAX_OUTPUT_TOKENS = 256
RESIZE_MAX = 640  # 长边缩放上限，与 YOLO 输入一致；设为 0 禁用缩放

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"

PROMPT = (
    "判断图中是否有明显积水。"
    "只返回 JSON，不要多余文字："
    '{"is_flood": true或false, "confidence": 0到1之间的小数}'
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_ollama_session() -> requests.Session:
    """访问本地 Ollama 时绕过系统代理，避免 localhost 请求被代理转发。"""
    session = requests.Session()
    session.trust_env = False
    return session


OLLAMA_SESSION = build_ollama_session()


def ensure_ollama_running() -> bool:
    """检查 ollama 服务是否可用，必要时尝试启动。"""
    try:
        response = OLLAMA_SESSION.get(OLLAMA_TAGS_URL, timeout=3)
        if response.status_code == 200:
            print("ollama 服务已在运行")
            return True
    except Exception:
        pass

    print("ollama 未运行，正在尝试启动...")
    ollama_executable = shutil.which("ollama")
    if ollama_executable is None:
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            ollama_executable = str(candidate)
    if ollama_executable is None:
        print("⚠ 未找到 ollama 可执行文件")
        return False

    subprocess.Popen(
        [ollama_executable, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(15):
        time.sleep(1)
        try:
            response = OLLAMA_SESSION.get(OLLAMA_TAGS_URL, timeout=2)
            if response.status_code == 200:
                print("ollama 服务已启动")
                return True
        except Exception:
            pass

    print("⚠ ollama 服务启动失败，请手动运行: ollama serve")
    return False


def ensure_model_available(model_name: str) -> bool:
    """确认目标模型已在 Ollama 中可用。"""
    try:
        response = OLLAMA_SESSION.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = {model.get("name") for model in models}
        # 同时匹配带 tag 和不带 tag 的模型名 (如 "model:latest" 匹配 "model")
        base_names = {name.rsplit(":", 1)[0] for name in model_names if name}
        if model_name in model_names or model_name in base_names:
            return True
    except Exception as exc:
        print(f"⚠ 无法检查模型列表: {exc}")
        return False

    print(f"⚠ 未找到模型: {model_name}")
    return False


def discover_images(image_dir: Path, max_images: int) -> list[Path]:
    """从目录中按文件名顺序选取图片。"""
    if not image_dir.exists():
        return []
    images = [
        path for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images[:max_images]


def image_to_base64(image_path: Path) -> str:
    """读取图片，按 RESIZE_MAX 缩放后返回 base64（不修改原文件）。"""
    import base64
    from io import BytesIO

    if RESIZE_MAX > 0:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) > RESIZE_MAX:
                scale = RESIZE_MAX / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def get_image_resolution(image_path: Path) -> tuple[int, int]:
    """返回 (width, height)。"""
    with Image.open(image_path) as img:
        return img.size


def get_memory_mb() -> float | None:
    """返回当前进程 RSS 内存占用（MB），psutil 不可用时返回 None。"""
    if psutil is None:
        return None
    return round(psutil.Process().memory_info().rss / 1024 / 1024, 2)


def parse_response_content(content: str) -> dict:
    """尽量把模型输出解析成 JSON。"""
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
    return {"is_flood": None, "confidence": 0.0, "reason": content}


def write_json(file_path: Path, data: object) -> None:
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * ratio
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def benchmark_image(image_path: Path, raw_dir: Path, raw_prefix: str = "") -> dict:
    """对单张图片发起一次请求并返回性能指标。"""
    width, height = get_image_resolution(image_path)
    mem_before = get_memory_mb()

    options = {
        "temperature": 0.0,
        "num_predict": MAX_OUTPUT_TOKENS,
    }
    if FORCE_CPU:
        options["num_gpu"] = 0

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_to_base64(image_path)],
            }
        ],
        "stream": False,
        "think": False,
        "options": options,
    }

    start = time.perf_counter()
    response = OLLAMA_SESSION.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
    wall_seconds = time.perf_counter() - start
    response.raise_for_status()

    mem_after = get_memory_mb()
    raw_data = response.json()
    content = raw_data.get("message", {}).get("content", "").strip()
    thinking = raw_data.get("message", {}).get("thinking", "").strip()
    parsed = parse_response_content(content)

    raw_file = raw_dir / f"{raw_prefix}{image_path.stem}.json"
    write_json(raw_file, raw_data)

    return {
        "image": image_path.name,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "image_size_mb": round(image_path.stat().st_size / 1024 / 1024, 4),
        "wall_seconds": round(wall_seconds, 4),
        "total_seconds": round(raw_data.get("total_duration", 0) / 1e9, 4),
        "load_seconds": round(raw_data.get("load_duration", 0) / 1e9, 4),
        "prompt_eval_seconds": round(raw_data.get("prompt_eval_duration", 0) / 1e9, 4),
        "eval_seconds": round(raw_data.get("eval_duration", 0) / 1e9, 4),
        "prompt_tokens": raw_data.get("prompt_eval_count", 0),
        "output_tokens": raw_data.get("eval_count", 0),
        "is_flood": parsed.get("is_flood"),
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason"),
        "thinking": thinking[:200] if thinking else "",
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "raw_file": str(raw_file),
    }


def append_jsonl(file_path: Path, row: dict) -> None:
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(file_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with file_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], run_dir: Path) -> dict:
    wall = sorted(row["wall_seconds"] for row in rows)
    total = sorted(row["total_seconds"] for row in rows)
    loads = sorted(row["load_seconds"] for row in rows)
    mem_values = [row["mem_after_mb"] for row in rows if row.get("mem_after_mb") is not None]
    summary = {
        "model": VLM_MODEL,
        "image_count": len(rows),
        "output_dir": str(run_dir),
        "wall_seconds": {
            "avg": round(statistics.mean(wall), 4),
            "min": round(min(wall), 4),
            "max": round(max(wall), 4),
            "p50": round(percentile(wall, 0.5), 4),
            "p95": round(percentile(wall, 0.95), 4),
        },
        "total_seconds": {
            "avg": round(statistics.mean(total), 4),
            "min": round(min(total), 4),
            "max": round(max(total), 4),
            "p50": round(percentile(total, 0.5), 4),
            "p95": round(percentile(total, 0.95), 4),
        },
        "load_seconds": {
            "avg": round(statistics.mean(loads), 4),
            "min": round(min(loads), 4),
            "max": round(max(loads), 4),
        },
        "memory_mb": {
            "max": round(max(mem_values), 2),
            "min": round(min(mem_values), 2),
            "avg": round(statistics.mean(mem_values), 2),
        } if mem_values else None,
        "flood_true_count": sum(1 for row in rows if row["is_flood"] is True),
        "flood_false_count": sum(1 for row in rows if row["is_flood"] is False),
        "flood_unknown_count": sum(1 for row in rows if row["is_flood"] is None),
    }
    return summary


def main() -> None:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_name
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("VLM 批量图片性能测试")
    print("=" * 72)
    print(f"模型: {VLM_MODEL}")
    print(f"强制 CPU: {FORCE_CPU}")
    print(f"默认图片目录: {DEFAULT_IMAGE_DIR}")
    print(f"输出目录: {run_dir}")

    if not ensure_ollama_running():
        return
    if not ensure_model_available(VLM_MODEL):
        return

    images = discover_images(DEFAULT_IMAGE_DIR, MAX_IMAGES)
    if not images:
        print("⚠ 没有找到可测试图片")
        return

    (run_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    (run_dir / "images.txt").write_text("\n".join(str(path) for path in images), encoding="utf-8")

    print(f"图片数量: {len(images)}")
    if WARMUP_FIRST:
        print("预热: 先用首张图预热一次，预热结果不计入最终统计")
        try:
            warmup = benchmark_image(images[0], raw_dir, raw_prefix="warmup_")
            warmup["phase"] = "warmup"
            append_jsonl(run_dir / "results.jsonl", warmup)
            print(
                f"预热完成 | {images[0].name} | wall={warmup['wall_seconds']:.2f}s | "
                f"reason={warmup['reason']}"
            )
        except Exception as exc:
            print(f"预热失败，继续正式测试: {exc}")

    rows: list[dict] = []
    for index, image_path in enumerate(images, start=1):
        print("-" * 72)
        print(f"[{index:02d}/{len(images):02d}] {image_path.name}")
        try:
            row = benchmark_image(image_path, raw_dir)
            row["phase"] = "benchmark"
            rows.append(row)
            append_jsonl(run_dir / "results.jsonl", row)
            write_csv(run_dir / "results.csv", rows)
            print(
                f"wall={row['wall_seconds']:.2f}s | total={row['total_seconds']:.2f}s | "
                f"load={row['load_seconds']:.2f}s | is_flood={row['is_flood']} | "
                f"reason={row['reason']}"
            )
        except Exception as exc:
            error_row = {
                "image": image_path.name,
                "image_path": str(image_path),
                "image_width": None,
                "image_height": None,
                "image_size_mb": round(image_path.stat().st_size / 1024 / 1024, 4),
                "wall_seconds": None,
                "total_seconds": None,
                "load_seconds": None,
                "prompt_eval_seconds": None,
                "eval_seconds": None,
                "prompt_tokens": None,
                "output_tokens": None,
                "is_flood": None,
                "confidence": None,
                "reason": f"ERROR: {exc}",
                "thinking": "",
                "mem_before_mb": None,
                "mem_after_mb": None,
                "raw_file": "",
                "phase": "benchmark",
            }
            rows.append(error_row)
            append_jsonl(run_dir / "results.jsonl", error_row)
            write_csv(run_dir / "results.csv", rows)
            print(f"ERROR: {exc}")

    ok_rows = [row for row in rows if isinstance(row["wall_seconds"], float)]
    if not ok_rows:
        print("⚠ 没有成功结果，无法生成 summary")
        return

    summary = summarize(ok_rows, run_dir)
    write_json(run_dir / "summary.json", summary)

    print("=" * 72)
    print("测试完成")
    print(f"成功图片: {len(ok_rows)} / {len(rows)}")
    print(f"wall 平均耗时: {summary['wall_seconds']['avg']:.2f}s")
    print(f"wall P95: {summary['wall_seconds']['p95']:.2f}s")
    print(f"wall 最大值: {summary['wall_seconds']['max']:.2f}s")
    if summary.get("memory_mb"):
        print(f"进程内存: avg={summary['memory_mb']['avg']:.1f}MB  max={summary['memory_mb']['max']:.1f}MB")
    print(f"结果目录: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()