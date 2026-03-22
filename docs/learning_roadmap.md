# YOLO 物体检测 — 完整学习路线图

> 面向有 Java 后端经验、零 Python/AI 基础的工程师
> 目标：从 Hello World 到能在实际项目中应用物体检测

---

## 学习大纲

```
阶段一: 概念理解     → 知道 YOLO 是什么，解决什么问题
阶段二: 快速上手     → 跑通 demo，感受效果              ← 你现在在这里
阶段三: 深入理解     → 理解模型参数、检测结果的含义
阶段四: 自定义训练   → 用自己的数据训练模型
阶段五: 工程化部署   → 与 Java 后端集成，做成可用的服务
阶段六: 持续精进     → 优化精度、速度，了解前沿动态
```

---

## 阶段一：概念理解（必读）

### 1.1 什么是物体检测（Object Detection）？

| 任务类型 | 输入 | 输出 | 类比 |
|---------|------|------|------|
| **图片分类** | 一张图 | 这张图是什么（1个类别） | SQL: `SELECT category FROM images WHERE id = ?` |
| **物体检测** | 一张图 | 图中有哪些物体、在哪里（N个框+类别） | SQL: `SELECT name, x, y, w, h FROM objects WHERE image_id = ?` |
| **语义分割** | 一张图 | 每个像素属于什么类别 | 给图片中每个像素"贴标签" |
| **实例分割** | 一张图 | 每个物体的精确轮廓 | 物体检测 + 语义分割的结合 |

YOLO 主要解决的是**物体检测**问题：输入一张图片，输出图中所有物体的位置（矩形框）和类别。

### 1.2 YOLO 发展简史

```
2015  YOLOv1  — 开创性工作，提出"一次看完"的理念
2016  YOLOv2  — 引入 Batch Normalization，精度提升
2018  YOLOv3  — 多尺度检测，实用性大增
2020  YOLOv4  — CSPDarknet 骨干网络
2020  YOLOv5  — Ultralytics 公司开发，PyTorch 实现，易用性飞跃 ★
2022  YOLOv6  — 美团开发，面向工业部署
2022  YOLOv7  — 精度和速度的新平衡
2023  YOLOv8  — Ultralytics 最新力作，API 极简，生态完善 ★★★ ← 我们学的
2024  YOLOv9  — 引入 PGI（可编程梯度信息）
2024  YOLOv10 — 引入 NMS-free 检测
2024  YOLO11  — Ultralytics 最新版（注意不叫 v11）
```

> **选择建议**：入门学 YOLOv8，API 最简洁、文档最全、社区最活跃。掌握后可无缝切换到 YOLO11。

### 1.3 核心概念速查表

| 术语 | 解释 | Java 类比 |
|------|------|-----------|
| **模型 (Model)** | 经过训练的"判断器"，能识别特定类型的物体 | 一个已部署的微服务 |
| **权重 (.pt文件)** | 模型学到的"知识"，以数字矩阵形式存储 | 数据库中的数据 |
| **推理 (Inference)** | 用模型处理新图片/视频，得到检测结果 | 调用 API 接口 |
| **训练 (Training)** | 用大量标注数据"教"模型识别物体 | 编写并测试代码直到通过 |
| **预训练模型** | 已在大数据集上训练好的模型，可直接使用 | 开源 SDK / JAR 依赖 |
| **微调 (Fine-tuning)** | 在预训练基础上，用少量自己的数据继续训练 | 在开源项目上做定制开发 |
| **数据集 (Dataset)** | 一组带标注的图片（图片 + 标签文件） | 测试用例集 |
| **标注 (Annotation)** | 在图片上画框，标记物体位置和类别 | 编写测试预期结果 |
| **置信度 (Confidence)** | 0~1 的分数，模型对检测结果的确信程度 | 搜索引擎相关性分数 |
| **IoU (交并比)** | 预测框和真实框的重叠程度，评估精度 | 单元测试覆盖率 |
| **mAP (平均精度)** | 综合评价模型检测能力的指标（0~1，越大越好） | 单元测试通过率 |
| **Epoch (轮次)** | 训练时遍历整个数据集一次 | 一次完整的 CI/CD 流水线 |
| **Batch Size** | 每次训练同时处理的图片数量 | 线程池同时处理的任务数 |
| **NMS (非极大值抑制)** | 去除重叠的检测框，保留最好的 | 去重操作 |
| **Backbone (骨干网络)** | 模型中负责提取图像特征的部分 | 底层通用框架 (Spring) |
| **Head (检测头)** | 模型中负责输出检测结果的部分 | 业务逻辑层 |

### 1.4 YOLO 怎么工作的？（简化版）

```
传入图片 (640x640)
    │
    ▼
┌─────────────┐
│  Backbone   │  提取图像特征（"看到了什么元素"）
│  骨干网络    │  类比：把原始请求解析成结构化数据
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Neck     │  融合不同尺度的特征（大物体+小物体）
│  颈部网络    │  类比：聚合来自多个数据源的信息
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Head     │  输出检测结果：每个物体的位置+类别+置信度
│   检测头     │  类比：生成最终的响应结果
└──────┬──────┘
       │
       ▼
  检测结果: [
    {class: "person", conf: 0.95, box: [100, 200, 300, 500]},
    {class: "car",    conf: 0.87, box: [400, 300, 700, 550]},
  ]
```

### 1.5 预训练模型能检测什么？

YOLOv8 预训练模型基于 **COCO 数据集**训练，可检测 **80 类**常见物体：

```
人和动物: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
交通工具: bicycle, car, motorcycle, airplane, bus, train, truck, boat
食物相关: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
家具家电: chair, couch, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard,
          cell phone, microwave, oven, toaster, sink, refrigerator
运动器材: frisbee, skis, snowboard, sports ball, kite, baseball bat/glove, skateboard,
          surfboard, tennis racket
日常用品: bottle, wine glass, cup, fork, knife, spoon, bowl, umbrella, handbag, tie,
          suitcase, backpack
其他:     traffic light, fire hydrant, stop sign, parking meter, bench, potted plant,
          book, clock, scissors, teddy bear, hair drier, toothbrush, vase
```

> 如果你要检测**不在以上列表**中的物体，就需要自己训练模型（见阶段四）。

---

## 阶段二：快速上手 ← 你现在在这里

### 2.1 本项目脚本使用顺序

| 序号 | 脚本 | 做什么 | 预期耗时 |
|------|------|--------|---------|
| 1 | `scripts/01_basic_detect.py` | 图片检测 Hello World | 1分钟 |
| 2 | `scripts/02_video_detect.py` | 视频文件检测 | 3分钟 |
| 3 | `scripts/03_webcam_detect.py` | 摄像头实时检测 | 随时 |
| 4 | `scripts/04_custom_train.py` | 自定义训练入门 | 5分钟 |

### 2.2 理解脚本中的关键 API

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")            # 从预训练权重加载
model = YOLO("yolov8n.yaml")          # 从配置文件创建新模型（需要自己训练）
model = YOLO("runs/train/best.pt")    # 加载自己训练的模型

# 推理（检测）
results = model.predict(source, conf=0.5, save=True)
# source 可以是: 图片路径、视频路径、URL、numpy数组、摄像头编号(0)

# 训练
model.train(data="coco128.yaml", epochs=100, imgsz=640)

# 验证
metrics = model.val(data="coco128.yaml")

# 导出（部署用）
model.export(format="onnx")           # 导出为 ONNX 格式
model.export(format="engine")         # 导出为 TensorRT（最快）
```

---

## 阶段三：深入理解

### 3.1 模型大小选择

| 模型 | 参数量 | 速度(A100) | mAP50-95 | 适用场景 |
|------|--------|-----------|----------|---------|
| YOLOv8n | 3.2M | 1.2ms | 37.3 | 边缘设备、实时应用、入门 |
| YOLOv8s | 11.2M | 1.8ms | 44.9 | 轻量级部署 |
| YOLOv8m | 25.9M | 5.0ms | 50.2 | 精度与速度平衡 |
| YOLOv8l | 43.7M | 7.1ms | 52.9 | 高精度要求 |
| YOLOv8x | 68.2M | 10.7ms | 53.9 | 最高精度、服务器部署 |

> **选择原则**：先用 n 跑通，确认效果后再换大模型。类似 Java 性能优化——先让它跑起来，再优化。

### 3.2 检测结果数据结构

```python
results = model.predict("image.jpg")
result = results[0]                    # 第一张图的结果

# 检测框信息
result.boxes.xyxy    # 框坐标 [N, 4]  (左上x, 左上y, 右下x, 右下y)
result.boxes.xywh    # 框坐标 [N, 4]  (中心x, 中心y, 宽, 高)
result.boxes.conf    # 置信度 [N]      0~1
result.boxes.cls     # 类别编号 [N]
result.names         # {0: 'person', 1: 'bicycle', ...} 类别映射

# 可视化
result.plot()        # 返回带标注的图片（numpy.ndarray）
result.show()        # 弹窗显示
result.save("out.jpg")  # 保存到文件
```

### 3.3 关键参数详解

```python
model.predict(
    source="image.jpg",
    conf=0.25,       # 置信度阈值，低于此值的检测结果被丢弃
    iou=0.7,         # NMS 的 IoU 阈值，重叠度>此值的框会被合并
    imgsz=640,       # 输入图片尺寸，越大越准但越慢
    half=True,       # 半精度推理（FP16），速度翻倍，精度几乎不变
    device=0,        # 使用的设备：0=GPU, 'cpu'=CPU
    classes=[0, 2],  # 只检测特定类别（0=person, 2=car）
    max_det=300,     # 每张图最多检测多少个物体
    augment=True,    # 测试时增强（TTA），更准但更慢
    verbose=False,   # 是否打印详细日志
)
```

---

## 阶段四：自定义训练

### 4.1 完整训练流程

```
第1步: 收集数据 → 拍照/截图/从网上收集目标物体的图片（最少 100 张起步）
第2步: 数据标注 → 用标注工具在图片上画框，标记物体类别
第3步: 组织数据 → 按 YOLO 格式组织目录结构
第4步: 编写配置 → 创建 data.yaml 描述数据集
第5步: 开始训练 → model.train(data="data.yaml", epochs=100)
第6步: 评估验证 → model.val() 查看模型精度
第7步: 迭代优化 → 根据结果调整数据/参数/模型大小
```

### 4.2 数据集目录结构

```
my_dataset/
├── train/                 # 训练集（80% 的数据）
│   ├── images/            # 图片
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── labels/            # 标注文件（与图片一一对应）
│       ├── img001.txt
│       └── img002.txt
├── val/                   # 验证集（20% 的数据）
│   ├── images/
│   └── labels/
└── data.yaml              # 数据集配置文件
```

### 4.3 标注文件格式

每张图片对应一个 `.txt` 文件，每行一个物体：
```
<class_id> <center_x> <center_y> <width> <height>
```
坐标值都是 0~1 的相对比例值。例如：
```
0 0.5 0.4 0.3 0.6     # 类别0的物体，中心在(50%, 40%)，宽30%，高60%
2 0.2 0.7 0.15 0.25   # 类别2的物体
```

### 4.4 data.yaml 配置文件

```yaml
# my_dataset/data.yaml
path: /path/to/my_dataset    # 数据集根目录
train: train/images           # 训练图片目录（相对于 path）
val: val/images               # 验证图片目录

# 类别定义
names:
  0: defect_crack             # 裂纹缺陷
  1: defect_scratch           # 划痕缺陷
  2: normal                   # 正常
```

### 4.5 推荐标注工具

| 工具 | 特点 | 推荐场景 |
|------|------|---------|
| **LabelImg** | 桌面应用，简单直观 | 快速标注少量图片 |
| **Roboflow** | 在线平台，自动增强 | 个人项目，数据集管理 |
| **CVAT** | 开源在线平台，功能全面 | 团队协作 |
| **Label Studio** | 通用标注平台 | 多种标注任务 |

---

## 阶段五：工程化部署

### 5.1 模型导出格式

| 格式 | 命令 | 适用场景 | 推理速度 |
|------|------|---------|---------|
| PyTorch (.pt) | 默认 | 开发调试 | 基准 |
| ONNX (.onnx) | `model.export(format="onnx")` | **Java 集成** ★ | 1.5x |
| TensorRT (.engine) | `model.export(format="engine")` | NVIDIA GPU 部署 | 3~5x |
| OpenVINO | `model.export(format="openvino")` | Intel CPU/GPU | 2~3x |
| CoreML | `model.export(format="coreml")` | Apple 设备 | - |

### 5.2 Java 集成方案（你最关心的）

#### 方案 A：Java 调用 ONNX 模型（推荐）
```
架构: Java (Spring Boot) + ONNX Runtime (Java SDK)

步骤:
1. Python 端: model.export(format="onnx")  → 得到 best.onnx
2. Java 端: 引入 onnxruntime 依赖
3. Java 端: 加载模型 → 预处理图片 → 推理 → 后处理结果

优点: 纯 Java 运行，无需 Python 环境
缺点: 需要自己写前后处理代码
```

```xml
<!-- Maven 依赖 -->
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.17.0</version>
</dependency>
<!-- GPU 加速版本 -->
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime_gpu</artifactId>
    <version>1.17.0</version>
</dependency>
```

#### 方案 B：Java 调用 Python 服务（快速）
```
架构: Java (Spring Boot) → HTTP → Python (FastAPI + YOLO)

步骤:
1. Python 端: 用 FastAPI 包装 YOLO 推理为 REST API
2. Java 端: 通过 HTTP 调用 Python 服务

优点: 实现最快，Python 端代码简单
缺点: 多一个 Python 服务需要维护
```

#### 方案 C：深度学习 Java 库（DJL）
```
架构: Java (Spring Boot + DJL)

步骤:
1. 导出 ONNX 或 TorchScript 模型
2. 用 DJL (Deep Java Library, Amazon 出品) 加载和推理

优点: 全 Java 生态，AWS 官方支持
缺点: 社区相对小，文档较少
```

### 5.3 部署架构建议

```
生产环境推荐架构:

┌──────────┐     ┌──────────────┐     ┌─────────────┐
│  前端/   │     │  Java 后端    │     │  Python     │
│  客户端   │────→│  (Spring Boot)│────→│  推理服务    │
│          │     │  业务逻辑     │     │  (FastAPI)  │
└──────────┘     └──────────────┘     └──────┬──────┘
                                             │
                                      ┌──────┴──────┐
                                      │  GPU 服务器  │
                                      │  (RTX/T4/A10)│
                                      └─────────────┘

QPS 要求不高时的精简架构:

┌──────────┐     ┌──────────────────┐
│  前端/   │     │  Java 后端        │
│  客户端   │────→│  (Spring Boot     │
│          │     │   + ONNX Runtime) │
└──────────┘     └──────────────────┘
                  纯 Java，无需 Python
```

---

## 阶段六：持续精进

### 6.1 提高检测精度的方法

1. **增加训练数据**：最直接有效，正确标注的数据越多越好
2. **数据增强**：旋转、缩放、颜色变换等（Ultralytics 自动做了）
3. **用更大的模型**：n → s → m → l → x
4. **增加训练轮数**：从 50 → 100 → 300 epochs
5. **调整超参数**：学习率、batch size、图片尺寸
6. **负样本**：加入不包含目标物体的图片，减少误检

### 6.2 提高推理速度的方法

1. **用更小的模型**：x → l → m → s → n
2. **半精度推理**：`model.predict(half=True)`
3. **导出 TensorRT**：`model.export(format="engine")`
4. **降低输入分辨率**：640 → 480 → 320
5. **批量推理**：一次处理多张图片

### 6.3 推荐学习资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 官方文档 | https://docs.ultralytics.com | 最权威，API 参考 |
| GitHub | https://github.com/ultralytics/ultralytics | 源码 + Issues |
| 视频课 | B站搜索 "YOLOv8 教程" | 中文视频教程 |
| 论文 | 搜索 "YOLOv8 paper" | 理解原理（可选） |
| 数据集 | https://roboflow.com/universe | 海量公开数据集 |
| 竞赛 | Kaggle 物体检测比赛 | 实战练习 |

### 6.4 技术路线扩展

```
                    ┌─── 图像分类 (cls)
                    │
物体检测 (detect) ──┼─── 实例分割 (segment)    ← YOLO 同样支持
                    │
                    ├─── 姿态估计 (pose)       ← YOLOv8-pose
                    │
                    ├─── 旋转框检测 (obb)       ← 适合遥感/航拍
                    │
                    └─── 目标跟踪 (track)       ← model.track()
```

YOLOv8 对以上任务都有统一的 API，学会了检测，切换其他任务只需改一个参数。

---

## 常见问题

### Q: CUDA out of memory
A: 减小 batch_size（如 8→4→2），或减小 imgsz（如 640→480）

### Q: 训练后精度很低
A: 检查标注质量、增加数据量、增加训练轮数。确保训练/验证集没有数据泄漏。

### Q: 推理速度不够快
A: 换小模型、用 TensorRT 导出、降低分辨率、开启 half 精度。

### Q: 在 Java 中怎么用
A: 导出 ONNX → 用 ONNX Runtime Java SDK 加载推理。详见阶段五。

---

*文档更新时间: 2026-03-22*
*Ultralytics 版本: 8.4.24*
