from ultralytics import YOLO

# 加载模型 (自动下载权重)
model = YOLO('yolov8n.pt') 

if __name__ == '__main__':
    # 开始训练
    model.train(
        data='dataset/data.yaml',  # 确保这个路径正确指向你的yaml
        epochs=100,
        imgsz=640,
        batch=16,                  # 4050 显存通常为 6GB，16 是个非常稳妥的数值
        device='0',                # 指定使用显卡
        workers=4,                 # 适合笔记本的并行数据读取数量
        amp=True                   # 开启自动混合精度，进一步提升显卡效率
    )
