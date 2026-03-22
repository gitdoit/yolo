# YOLOv8 物体检测 — 从零入门学习项目

> 面向有 Java 后端经验、零 Python/AI 基础的工程师，从环境搭建到实际物体检测的完整学习路径。

## 环境信息

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 (conda: yolov8-py310) |
| PyTorch | 2.10.0+cu126 |
| Ultralytics | 8.4.24 |
| CUDA | 12.6 |
| GPU | NVIDIA GeForce RTX 4070 SUPER (12GB) |
| OS | Windows |

## 项目结构

```
yolo_learn/
├── data/
│   ├── images/              # 测试图片
│   └── videos/              # 测试视频
├── models/                  # 模型权重文件（自动下载）
├── outputs/                 # 检测结果输出
├── scripts/                 # 学习脚本（按编号递进）
│   ├── 01_basic_detect.py   # ★ Hello World：图片物体检测
│   ├── 02_video_detect.py   # 视频文件检测
│   ├── 03_webcam_detect.py  # 摄像头实时检测
│   └── 04_custom_train.py   # 自定义训练入门
├── docs/
│   └── learning_roadmap.md  # 完整学习路线图
├── requirements.txt         # 依赖清单
├── .gitignore
└── README.md                # 本文件
```

## 快速开始

### 1. 激活环境
```bash
conda activate yolov8-py310
```

### 2. 运行 Hello World（图片检测）
```bash
cd e:\workspace\python\yolo_learn
python scripts/01_basic_detect.py
```
首次运行会自动下载 yolov8n.pt 模型（约 6MB），检测结果保存在 `outputs/` 目录。

### 3. 视频检测
```bash
python scripts/02_video_detect.py
```

### 4. 摄像头实时检测
```bash
python scripts/03_webcam_detect.py
```
按 `q` 键退出。

## 核心概念速览（Java 工程师视角）

| YOLO 概念 | Java 类比 | 说明 |
|-----------|-----------|------|
| `YOLO("yolov8n.pt")` | `new Service()` | 加载模型 = 初始化服务 |
| `model.predict()` | `service.process()` | 推理 = 调用业务方法 |
| 预训练模型 (.pt 文件) | JAR 依赖包 | 已训练好的模型权重，可直接使用 |
| `results[0].boxes` | `List<DetectionResult>` | 检测结果列表 |
| `conf` (置信度) | 匹配分数 | 0~1，表示模型对检测结果的确信程度 |
| `cls` (类别) | 分类枚举 | 检测到的物体类别编号 |
| 训练 (train) | 编译+部署 | 用数据"教会"模型识别特定物体 |
| 推理 (predict/inference) | 运行时调用 | 用训练好的模型进行实际检测 |

## 详细学习路线

参见 [docs/learning_roadmap.md](docs/learning_roadmap.md)
