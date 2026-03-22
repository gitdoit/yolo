from ultralytics import YOLO

yolo = YOLO("./models/yolov8n.pt",task="detect")

result = yolo(source="screen")  # 加载预训练模型pyu