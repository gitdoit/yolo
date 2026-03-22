"""下载 yolov8n.pt 模型文件，尝试多个镜像源"""
import urllib.request
import os
import sys

MIRRORS = [
    "https://mirror.ghproxy.com/github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "https://hub.gitmirror.com/github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "https://gh-proxy.com/github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
]

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolov8n.pt")

# 清理可能存在的不完整文件
if os.path.exists(target):
    os.remove(target)

for url in MIRRORS:
    try:
        print(f"尝试下载: {url[:60]}...")
        urllib.request.urlretrieve(url, target)
        size_mb = os.path.getsize(target) / 1024 / 1024
        print(f"下载完成! 文件大小: {size_mb:.1f} MB")
        if size_mb > 5:
            print("文件大小合理，下载成功!")
            sys.exit(0)
        else:
            print("文件太小，可能不完整，尝试下一个镜像...")
            os.remove(target)
    except Exception as e:
        print(f"失败: {e}")
        if os.path.exists(target):
            os.remove(target)
        continue

print("所有镜像均失败，请手动下载:")
print("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt")
print(f"下载后放到: {os.path.dirname(target)}")
