from ultralytics import YOLO
import cv2

# 1. load the model has been trained
model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)


cap = cv2.VideoCapture(0)

print("opening camera，press 'q' to exit...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

   #frame detection
    results = model(frame, conf=0.5) 

    annotated_frame = results[0].plot()
  
    cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
