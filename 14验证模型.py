from ultralytics import YOLO
import cv2

# 1. 加载你训练好的模型
model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

# 2. 开启摄像头 (0 代表默认摄像头)
cap = cv2.VideoCapture(0)

print("正在启动摄像头，按 'q' 键退出...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. 对当前帧进行检测
    results = model(frame, conf=0.5)  # conf=0.1 表示置信度超过50%才显示

    # 4. 在图像上绘制识别结果
    annotated_frame = results[0].plot()

    # 5. 显示画面
    cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()