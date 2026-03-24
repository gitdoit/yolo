# 07 语义分割动态 ROI 交通拥堵检测 — 学习总结

## 一、什么是语义分割

**语义分割 = 给图像中每一个像素分配一个类别标签。**

### 与其他视觉任务的对比

| 任务 | 输出 | 粒度 |
|------|------|------|
| 图像分类 | 整张图: "这是街景" | 最粗 — 一张图一个标签 |
| 目标检测 (YOLO) | 矩形框: "这里有辆车" | 中等 — 每个物体一个框 |
| 语义分割 | 逐像素: "这个像素是道路" | 最细 — 每个像素一个标签 |

### 语义分割 vs 实例分割

- **语义分割**: 所有车都标为"车"类，不区分第 1 辆还是第 2 辆 → 适合回答"哪里是道路"
- **实例分割**: 不仅标类别，还区分每个个体（车1、车2、车3） → 适合回答"这辆车的精确轮廓"

本脚本中：
- 道路 → 用语义分割（只需知道"哪些像素是路"）
- 车辆 → 用目标检测框（只需面积占比，矩形框已够用，比实例分割更快）

---

## 二、为什么要做这个脚本（方案 B）

### 原方案（06 脚本）的问题：固定矩形 ROI

| 问题 | 说明 |
|------|------|
| PTZ 摄像头转动后 ROI 失效 | 手动框选的矩形区域不再对准道路 |
| 道路形状不规则 | 弯道、交叉口无法用矩形精确覆盖 |
| 每个摄像头都需人工标注 | 无法规模化部署 |

### 新方案（07 脚本）的解决

| 改进点 | 做法 |
|--------|------|
| **自动识别道路** | SegFormer 逐像素分类，无需人工标注 |
| **不规则形状** | 掩码跟随道路实际轮廓，非矩形 |
| **摄像头转动自适应** | 每 30 帧重新分割，动态更新道路掩码 |
| **精确密度计算** | 车辆框 ∩ 道路像素数，而非矩形交集 |

---

## 三、使用的技术栈

### 1. SegFormer-B0 (Cityscapes) — 道路语义分割

- **模型**: `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`
- **大小**: ~15MB，轻量级
- **架构**: Transformer，感受野大，对道路这种大面积连续区域效果好
- **训练数据**: Cityscapes 数据集（专门针对城市街景），支持 19 个类别
- **来源**: HuggingFace Transformers 库，首次运行自动下载

**Cityscapes 19 个类别：**

```
 0: road（道路） ← 我们用这个
 1: sidewalk（人行道）
 2: building（建筑）
 3: wall / 4: fence / 5: pole
 6: traffic light / 7: traffic sign
 8: vegetation / 9: terrain / 10: sky
11: person / 12: rider
13: car / 14: truck / 15: bus / 16: train
17: motorcycle / 18: bicycle
```

### 2. YOLOv8n — 车辆目标检测

- **模型**: `yolov8n.pt`（COCO 预训练）
- **检测类别**: car(2), motorcycle(3), bus(5), truck(7)
- **置信度阈值**: 0.35（交通场景适当降低以检测远处小车）

---

## 四、核心代码解析

### 数据流总览

```
摄像头帧
   │
   ├──→ SegFormer (每30帧) ──→ road_mask (道路二值掩码)
   │                                    │
   ├──→ YOLOv8 (每2帧) ──→ 车辆检测框 ──┤
   │                                    │
   │                              ∩ 交集计算
   │                                    │
   │                         车辆在道路上的面积占比
   │                                    │
   │                         滑动窗口平滑 (15帧)
   │                                    │
   │                         拥堵等级判定
   │                                    │
   └──→ 原始帧 + 道路着色 + 车辆框 + 信息面板 ──→ 显示
```

### 4.1 语义分割流水线

```
原始帧 → BGR转RGB → SegformerImageProcessor预处理 → 模型推理
→ logits → 双线性插值上采样到原图尺寸 → argmax取类别
→ 提取 road 类别 → 二值掩码 (255=道路, 0=非道路)
```

关键代码：
```python
def segment_road(frame, processor, model, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # SegFormer 输出尺寸是原图的 1/4，需要上采样
    h, w = frame.shape[:2]
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    # 在 19 个类别中取概率最大的
    pred = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    # 只保留道路像素
    road_mask = np.where(pred == ROAD_CLASS_ID, 255, 0).astype(np.uint8)
    return road_mask
```

### 4.2 基于掩码的密度计算

**与 06 脚本的本质区别：**
```
06: 车辆框面积 ∩ 矩形ROI面积 / 矩形ROI总面积
07: 车辆框内的道路像素数 / 道路总像素数
```

关键代码：
```python
# 截取检测框对应区域的道路掩码
box_road = road_mask[y1:y2, x1:x2]
# 统计框内有多少像素是道路
intersection = np.count_nonzero(box_road)
```

优势：一辆车停在路边，半个车身在人行道上，只会统计它在道路上的那部分面积。

### 4.3 帧率优化：双频率策略

| 操作 | 间隔 | 原因 |
|------|------|------|
| 道路分割 | 每 30 帧 | 道路不会每帧变化，分割计算量大 (~50ms/帧) |
| 车辆检测 | 每 2 帧 | 车辆移动较快，需要较高频率更新 |

中间帧复用上一次的缓存结果（`road_mask` 和 `last_detect`），大幅降低 GPU 负载。

### 4.4 滑动窗口平滑

```python
density_history = deque(maxlen=15)  # 保留最近 15 帧的密度值
avg = np.mean(density_history)       # 取平均值判定拥堵
```

解决问题：单帧密度值抖动大（某帧突然检测到大卡车密度飙升，下帧消失），用滑动平均消除噪声。

### 4.5 道路掩码可视化

```python
overlay[road_mask == 255] = (
    overlay[road_mask == 255] * 0.6
    + np.array(color) * 0.4
).astype(np.uint8)
```

只对道路像素做半透明着色（颜色随拥堵等级变化：绿/橙/深橙/红），非道路区域保持原样。

---

## 五、对比总结

| 对比项 | 06（固定 ROI） | 07（语义分割动态 ROI） |
|--------|---------------|----------------------|
| ROI 来源 | 手动框选矩形 | SegFormer 自动识别道路像素 |
| ROI 形状 | 矩形 | 任意形状（跟随实际道路轮廓） |
| PTZ 适应 | 摄像头转动后 ROI 失效 | 每 30 帧自动更新道路掩码 |
| 密度计算 | 车辆框 ∩ 矩形面积 | 车辆框 ∩ 道路像素数 |
| 额外依赖 | 无 | `transformers` 库 |
| GPU 开销 | 较低 | 稍高（多一个分割模型） |

---

## 六、运行方式

```bash
# 安装依赖
pip install transformers

# 运行
conda activate yolov8
python scripts/07_traffic_congestion_seg.py
```

首次运行会自动从 HuggingFace 下载 SegFormer-B0 模型（约 15MB）。

---

## 七、可调参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SEG_MODEL_NAME` | `nvidia/segformer-b0-...` | 分割模型，可换更大的 b1~b5 |
| `SEG_EVERY_N_FRAMES` | 30 | 道路分割更新频率，越小越准但越慢 |
| `DETECT_EVERY_N_FRAMES` | 2 | 车辆检测频率 |
| `CONFIDENCE_THRESHOLD` | 0.35 | 车辆检测置信度阈值 |
| `SMOOTHING_WINDOW` | 15 | 滑动窗口大小，越大越平滑 |
| `CONGESTION_THRESHOLDS` | 8%/20%/40% | 拥堵等级分界线 |
