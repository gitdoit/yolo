"""
=============================================================================
21_clip_flood_gui.py — CLIP 零样本洪水分类 图形化界面
=============================================================================

用途：
  - 拖拽或选择图片，实时显示洪水/正常判定结果
  - 复用 20_clip_flood_benchmark.py 的模型与提示词

【运行方式】
  conda activate yolov8
  python scripts/21_clip_flood_gui.py
"""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, ttk

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from PIL import Image, ImageTk

# ============================================================================
# 配置（与脚本 20 保持一致）
# ============================================================================

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

FLOOD_POSITIVE_TEXTS = [
    "a photo of a flooded street",
    "a photo of standing water on the road",
    "a photo of a flood with water covering the ground",
    "a photo of a river overflowing its banks",
    "a photo of flood water surrounding buildings",
    "a photo of a waterlogged area after heavy rain",
]

FLOOD_NEGATIVE_TEXTS = [
    "a photo of a normal dry street",
    "a photo of a clear road with no water",
    "a photo of a sunny day with dry ground",
    "a photo of a clean city street",
    "a photo of a normal river within its banks",
    "a photo of a dry construction site",
]

PREVIEW_SIZE = (420, 320)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================================
# 颜色
# ============================================================================

BG = "#1e1e2e"
CARD = "#2a2a3e"
ACCENT_FLOOD = "#ff5555"
ACCENT_NORMAL = "#50fa7b"
ACCENT_BLUE = "#8be9fd"
TEXT_PRIMARY = "#f8f8f2"
TEXT_MUTED = "#6272a4"
BORDER = "#44475a"


# ============================================================================
# CLIP 推理后端
# ============================================================================

class ClipBackend:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.device = "cpu"
        self.n_pos = len(FLOOD_POSITIVE_TEXTS)
        self.n_neg = len(FLOOD_NEGATIVE_TEXTS)

    def load(self, progress_cb=None):
        import open_clip

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if progress_cb:
            progress_cb(f"加载模型 {CLIP_MODEL} ({self.device})…")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=self.device
        )
        self.model.eval()

        if progress_cb:
            progress_cb("编码文字描述…")

        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        all_texts = FLOOD_POSITIVE_TEXTS + FLOOD_NEGATIVE_TEXTS
        tokens = tokenizer(all_texts).to(self.device)
        with torch.no_grad():
            tf = self.model.encode_text(tokens)
            self.text_features = tf / tf.norm(dim=-1, keepdim=True)

        if progress_cb:
            progress_cb("就绪")

    def infer(self, img_path: str) -> dict:
        pil_img = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            vf = self.model.encode_image(tensor)
            vf = vf / vf.norm(dim=-1, keepdim=True)
        elapsed = time.perf_counter() - t0

        sims = (vf @ self.text_features.T).squeeze(0).cpu().tolist()
        pos = sims[: self.n_pos]
        neg = sims[self.n_pos :]
        avg_pos = sum(pos) / self.n_pos
        avg_neg = sum(neg) / self.n_neg
        gap = avg_pos - avg_neg

        return {
            "is_flood": gap > 0,
            "avg_positive": avg_pos,
            "avg_negative": avg_neg,
            "confidence_gap": gap,
            "elapsed": elapsed,
            "pos_scores": pos,
            "neg_scores": neg,
        }


# ============================================================================
# GUI
# ============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CLIP 洪水识别")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.backend = ClipBackend()
        self._photo = None          # 保持对 ImageTk 的引用

        self._build_ui()
        self._enable_drop()         # 拖拽支持（需要 tkinterdnd2，可选）

        # 异步加载模型
        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=12, pady=8)

        # —— 顶部标题 ——
        title_frame = tk.Frame(self, bg=BG)
        title_frame.pack(fill="x", padx=16, pady=(14, 0))
        tk.Label(
            title_frame,
            text="CLIP  零样本洪水识别",
            bg=BG, fg=TEXT_PRIMARY,
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left")
        self.status_lbl = tk.Label(
            title_frame,
            text="⏳ 正在加载模型…",
            bg=BG, fg=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        self.status_lbl.pack(side="right", padx=4)

        # —— 主体：左（预览）+ 右（结果） ——
        body = tk.Frame(self, bg=BG)
        body.pack(padx=16, pady=10)

        # 左列：图片预览 + 按钮
        left = tk.Frame(body, bg=CARD, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        left.grid(row=0, column=0, padx=(0, 12), sticky="nsew")

        self.preview_lbl = tk.Label(
            left,
            text="拖拽图片到此处\n或点击下方按钮选择",
            bg=CARD, fg=TEXT_MUTED,
            width=PREVIEW_SIZE[0] // 8,
            height=PREVIEW_SIZE[1] // 16,
            font=("Segoe UI", 10),
        )
        self.preview_lbl.pack(padx=2, pady=2)

        btn_row = tk.Frame(left, bg=CARD)
        btn_row.pack(fill="x", padx=8, pady=(0, 8))

        self.open_btn = tk.Button(
            btn_row,
            text="📂  选择图片",
            bg="#6272a4", fg=TEXT_PRIMARY,
            activebackground="#7282b4", activeforeground=TEXT_PRIMARY,
            relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            command=self._pick_file,
        )
        self.open_btn.pack(fill="x")

        # 右列：判定结果卡片
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        # 判定标签（大字）
        verdict_card = tk.Frame(right, bg=CARD, bd=0,
                                highlightthickness=1, highlightbackground=BORDER)
        verdict_card.pack(fill="x", pady=(0, 10))

        self.verdict_lbl = tk.Label(
            verdict_card,
            text="—",
            bg=CARD, fg=TEXT_MUTED,
            font=("Segoe UI", 36, "bold"),
        )
        self.verdict_lbl.pack(pady=(18, 2))

        self.verdict_sub = tk.Label(
            verdict_card,
            text="等待图片…",
            bg=CARD, fg=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        self.verdict_sub.pack(pady=(0, 16))

        # 数值详情
        detail_card = tk.Frame(right, bg=CARD, bd=0,
                               highlightthickness=1, highlightbackground=BORDER)
        detail_card.pack(fill="both", expand=True)

        tk.Label(
            detail_card, text="详细分数",
            bg=CARD, fg=TEXT_MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(10, 4))

        self.detail_text = tk.Text(
            detail_card,
            bg=CARD, fg=TEXT_PRIMARY,
            font=("Consolas", 9),
            relief="flat", bd=0,
            width=32, height=14,
            state="disabled",
            cursor="arrow",
        )
        self.detail_text.pack(padx=12, pady=(0, 10))

        # —— 进度条（推理时显示） ——
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=300)
        self.progress.pack(pady=(0, 10))

    # ------------------------------------------------------------------
    # 拖拽（可选，需要 tkinterdnd2）
    # ------------------------------------------------------------------

    def _enable_drop(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.preview_lbl.drop_target_register(DND_FILES)
            self.preview_lbl.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass  # 没有 tkinterdnd2 时静默忽略，仍可用按钮

    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            self._run_infer(path)

    # ------------------------------------------------------------------
    # 文件选择
    # ------------------------------------------------------------------

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self._run_infer(path)

    # ------------------------------------------------------------------
    # 模型加载（线程）
    # ------------------------------------------------------------------

    def _load_model(self):
        try:
            self.backend.load(progress_cb=self._set_status)
            self.after(0, lambda: self.open_btn.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self._set_status(f"模型加载失败: {e}"))

    # ------------------------------------------------------------------
    # 推理（线程）
    # ------------------------------------------------------------------

    def _run_infer(self, path: str):
        if self.backend.model is None:
            self._set_status("模型尚未就绪，请稍候…")
            return

        self.open_btn.configure(state="disabled")
        self.progress.start(12)
        self._set_status("推理中…")
        self._show_preview(path)

        def worker():
            try:
                result = self.backend.infer(path)
                self.after(0, lambda: self._show_result(path, result))
            except Exception as e:
                self.after(0, lambda: self._set_status(f"错误: {e}"))
            finally:
                self.after(0, self._stop_progress)

        threading.Thread(target=worker, daemon=True).start()

    def _stop_progress(self):
        self.progress.stop()
        self.open_btn.configure(state="normal")

    # ------------------------------------------------------------------
    # 图片预览
    # ------------------------------------------------------------------

    def _show_preview(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.preview_lbl.configure(image=self._photo, text="", width=0, height=0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 显示结果
    # ------------------------------------------------------------------

    def _show_result(self, path: str, r: dict):
        is_flood = r["is_flood"]
        gap = r["confidence_gap"]
        elapsed = r["elapsed"]

        if is_flood:
            verdict_text = "🌊  FLOOD"
            verdict_color = ACCENT_FLOOD
            sub_text = f"置信度差  {gap:+.4f}  ｜  洪水/积水场景"
        else:
            verdict_text = "✅  NORMAL"
            verdict_color = ACCENT_NORMAL
            sub_text = f"置信度差  {gap:+.4f}  ｜  正常干燥场景"

        self.verdict_lbl.configure(text=verdict_text, fg=verdict_color)
        self.verdict_sub.configure(
            text=sub_text + f"  ｜  {elapsed*1000:.1f} ms",
            fg=TEXT_PRIMARY,
        )

        # 详细分数
        lines = [
            f"{'文件':<8}{Path(path).name}",
            "",
            f"{'avg_pos':<12}{r['avg_positive']:.4f}",
            f"{'avg_neg':<12}{r['avg_negative']:.4f}",
            f"{'gap':<12}{gap:+.4f}",
            f"{'耗时':<12}{elapsed*1000:.1f} ms",
            "",
            "— 正面描述分数 —",
        ]
        for txt, sc in zip(FLOOD_POSITIVE_TEXTS, r["pos_scores"]):
            label = txt.replace("a photo of ", "")[:22]
            lines.append(f"  {label:<22} {sc:.4f}")
        lines.append("")
        lines.append("— 负面描述分数 —")
        for txt, sc in zip(FLOOD_NEGATIVE_TEXTS, r["neg_scores"]):
            label = txt.replace("a photo of ", "")[:22]
            lines.append(f"  {label:<22} {sc:.4f}")

        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", "\n".join(lines))
        self.detail_text.configure(state="disabled")

        self._set_status(f"完成  ｜  设备: {self.backend.device}  ｜  模型: {CLIP_MODEL}")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_lbl.configure(text=msg))


# ============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()
